-- Migration: 007_retrieval_flow.sql
-- Description: Human Server Minimum Viable Retrieval & Fragment Recall Flow:
--   1. memory_retrievals table (tracks retrieval requests initiated by memory owners)
--   2. fragment_recall_responses table (temporary transport buffer for recalled plaintext from human nodes)
--   3. Secure RPCs:
--      - get_user_stored_memories: Lists active memories owned by caller
--      - initiate_memory_retrieval: Creates/reuses open retrieval request for an active memory
--      - get_pending_recalls_for_node: Returns open recall requests for caller node's memorized fragments
--      - submit_fragment_recall: Verifies hash, stores temporary response, retires assignment to 'recalled' (frees slot)
--      - get_retrieval_status_and_fragments: Fetches collected recalled fragments for memory owner
--      - complete_memory_retrieval: Marks retrieval complete and immediately PURGES temporary recall plaintext
--   4. Explicit privileges and row-level security

-- 1. Create memory_retrievals table
CREATE TABLE IF NOT EXISTS public.memory_retrievals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES public.memories(id) ON DELETE CASCADE,
    requester_user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'completed', 'cancelled')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_retrievals_memory ON public.memory_retrievals(memory_id);
CREATE INDEX IF NOT EXISTS idx_retrievals_requester ON public.memory_retrievals(requester_user_id);
CREATE INDEX IF NOT EXISTS idx_retrievals_status ON public.memory_retrievals(status);

-- 2. Create fragment_recall_responses table (TEMPORARY TRANSPORT BUFFER ONLY)
-- Plaintext resides here temporarily until the memory owner reconstructs the message
-- and marks retrieval completed, at which point all rows are immediately deleted.
CREATE TABLE IF NOT EXISTS public.fragment_recall_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    retrieval_id UUID NOT NULL REFERENCES public.memory_retrievals(id) ON DELETE CASCADE,
    assignment_id UUID NOT NULL REFERENCES public.fragment_assignments(id) ON DELETE CASCADE,
    fragment_id UUID NOT NULL REFERENCES public.memory_fragments(id) ON DELETE CASCADE,
    node_id UUID NOT NULL REFERENCES public.nodes(id) ON DELETE CASCADE,
    recalled_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '1 day'),
    CONSTRAINT uq_retrieval_fragment_node UNIQUE (retrieval_id, fragment_id, node_id)
);

CREATE INDEX IF NOT EXISTS idx_recall_responses_retrieval ON public.fragment_recall_responses(retrieval_id);
CREATE INDEX IF NOT EXISTS idx_recall_responses_node ON public.fragment_recall_responses(node_id);

-- 3. Enable RLS
ALTER TABLE public.memory_retrievals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fragment_recall_responses ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own retrievals" ON public.memory_retrievals;
CREATE POLICY "Users can view their own retrievals"
    ON public.memory_retrievals
    FOR SELECT
    USING (auth.uid() = requester_user_id);

DROP POLICY IF EXISTS "Users can insert their own retrievals" ON public.memory_retrievals;
CREATE POLICY "Users can insert their own retrievals"
    ON public.memory_retrievals
    FOR INSERT
    WITH CHECK (auth.uid() = requester_user_id);

DROP POLICY IF EXISTS "Nodes can view recall responses for their submissions" ON public.fragment_recall_responses;
CREATE POLICY "Nodes can view recall responses for their submissions"
    ON public.fragment_recall_responses
    FOR SELECT
    USING (node_id IN (SELECT id FROM public.nodes WHERE user_id = auth.uid()));

DROP POLICY IF EXISTS "Memory owners can view recall responses for their retrievals" ON public.fragment_recall_responses;
CREATE POLICY "Memory owners can view recall responses for their retrievals"
    ON public.fragment_recall_responses
    FOR SELECT
    USING (retrieval_id IN (SELECT id FROM public.memory_retrievals WHERE requester_user_id = auth.uid()));

-- 4. Secure RPC: get_user_stored_memories
CREATE OR REPLACE FUNCTION public.get_user_stored_memories()
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_user_id UUID;
    v_result JSONB;
BEGIN
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', m.id,
        'created_at', m.created_at,
        'expires_at', m.expires_at,
        'fragment_count', m.fragment_count,
        'status', m.status
    ) ORDER BY m.created_at DESC), '[]'::JSONB)
    INTO v_result
    FROM public.memories m
    WHERE m.owner_user_id = v_user_id AND m.status = 'active';

    RETURN v_result;
END;
$$;

-- 5. Secure RPC: initiate_memory_retrieval
CREATE OR REPLACE FUNCTION public.initiate_memory_retrieval(p_memory_id UUID)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_user_id UUID;
    v_memory RECORD;
    v_retrieval_id UUID;
    v_assigned_node_count INT;
BEGIN
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    -- Validate ownership and active status
    SELECT m.id, m.fragment_count, m.status
    INTO v_memory
    FROM public.memories m
    WHERE m.id = p_memory_id AND m.owner_user_id = v_user_id;

    IF v_memory.id IS NULL THEN
        RAISE EXCEPTION 'Memory not found or not owned by user';
    END IF;

    IF v_memory.status != 'active' THEN
        RAISE EXCEPTION 'Memory is not active';
    END IF;

    -- Check if an open retrieval already exists
    SELECT r.id INTO v_retrieval_id
    FROM public.memory_retrievals r
    WHERE r.memory_id = p_memory_id AND r.status = 'open'
    ORDER BY r.created_at DESC
    LIMIT 1;

    IF v_retrieval_id IS NULL THEN
        INSERT INTO public.memory_retrievals (
            memory_id,
            requester_user_id,
            status,
            created_at
        )
        VALUES (
            p_memory_id,
            v_user_id,
            'open',
            now()
        )
        RETURNING id INTO v_retrieval_id;
    END IF;

    -- Count distinct human nodes holding active fragments for this memory
    SELECT count(DISTINCT fa.node_id) INTO v_assigned_node_count
    FROM public.fragment_assignments fa
    JOIN public.memory_fragments mf ON fa.fragment_id = mf.id
    WHERE mf.memory_id = p_memory_id
      AND fa.status IN ('assigned', 'fulfilled', 'memorized');

    RETURN jsonb_build_object(
        'retrieval_id', v_retrieval_id,
        'memory_id', p_memory_id,
        'status', 'open',
        'fragment_count', v_memory.fragment_count,
        'holding_nodes_count', v_assigned_node_count
    );
END;
$$;

-- 6. Secure RPC: get_pending_recalls_for_node
CREATE OR REPLACE FUNCTION public.get_pending_recalls_for_node()
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_user_id UUID;
    v_node_id UUID;
    v_result JSONB;
BEGIN
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RETURN '[]'::JSONB;
    END IF;

    SELECT n.id INTO v_node_id FROM public.nodes n WHERE n.user_id = v_user_id LIMIT 1;
    IF v_node_id IS NULL THEN
        RETURN '[]'::JSONB;
    END IF;

    -- Query open retrievals that have active memorized fragments assigned to this node
    -- and where this node has not yet submitted a response
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'retrieval_id', mr.id,
        'assignment_id', fa.id,
        'fragment_id', mf.id,
        'memory_id', mf.memory_id,
        'sequence_number', mf.sequence_number,
        'size_bytes', mf.size_bytes,
        'expected_hash', mf.hash,
        'retrieval_created_at', mr.created_at
    ) ORDER BY mr.created_at ASC, mf.sequence_number ASC), '[]'::JSONB)
    INTO v_result
    FROM public.memory_retrievals mr
    JOIN public.memories m ON mr.memory_id = m.id
    JOIN public.memory_fragments mf ON m.id = mf.memory_id
    JOIN public.fragment_assignments fa ON mf.id = fa.fragment_id
    WHERE mr.status = 'open'
      AND fa.node_id = v_node_id
      AND fa.status IN ('assigned', 'fulfilled', 'memorized')
      AND NOT EXISTS (
          SELECT 1 FROM public.fragment_recall_responses frr
          WHERE frr.retrieval_id = mr.id
            AND frr.assignment_id = fa.id
      );

    RETURN v_result;
END;
$$;

-- 7. Secure RPC: submit_fragment_recall
CREATE OR REPLACE FUNCTION public.submit_fragment_recall(
    p_retrieval_id UUID,
    p_assignment_id UUID,
    p_recalled_text TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_user_id UUID;
    v_node_id UUID;
    v_assignment RECORD;
    v_retrieval RECORD;
    v_actual_hash TEXT;
    v_clean_text TEXT;
BEGIN
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    SELECT n.id INTO v_node_id FROM public.nodes n WHERE n.user_id = v_user_id LIMIT 1;
    IF v_node_id IS NULL THEN
        RAISE EXCEPTION 'Node not found for authenticated user';
    END IF;

    v_clean_text := trim(p_recalled_text);
    IF v_clean_text IS NULL OR length(v_clean_text) = 0 THEN
        RAISE EXCEPTION 'Recalled text cannot be empty';
    END IF;

    -- Verify open retrieval
    SELECT r.id, r.memory_id, r.status
    INTO v_retrieval
    FROM public.memory_retrievals r
    WHERE r.id = p_retrieval_id AND r.status = 'open';

    IF v_retrieval.id IS NULL THEN
        RAISE EXCEPTION 'Retrieval request not found or not open';
    END IF;

    -- Verify assignment belongs to this node and matches the retrieval's memory
    SELECT fa.id, fa.fragment_id, fa.status, mf.hash, mf.memory_id
    INTO v_assignment
    FROM public.fragment_assignments fa
    JOIN public.memory_fragments mf ON fa.fragment_id = mf.id
    WHERE fa.id = p_assignment_id AND fa.node_id = v_node_id;

    IF v_assignment.id IS NULL THEN
        RAISE EXCEPTION 'Assignment not found or does not belong to caller node';
    END IF;

    IF v_assignment.memory_id != v_retrieval.memory_id THEN
        RAISE EXCEPTION 'Assignment does not belong to the requested memory';
    END IF;

    -- Server-side authoritative SHA-256 hash verification
    v_actual_hash := encode(extensions.digest(v_clean_text, 'sha256'), 'hex');
    IF lower(v_actual_hash) != lower(v_assignment.hash) THEN
        RAISE EXCEPTION 'Recall hash verification failed. The entered plaintext does not match the memorized fragment.';
    END IF;

    -- Store temporary transport response (1-day expiry)
    INSERT INTO public.fragment_recall_responses (
        retrieval_id,
        assignment_id,
        fragment_id,
        node_id,
        recalled_text,
        created_at,
        expires_at
    )
    VALUES (
        p_retrieval_id,
        p_assignment_id,
        v_assignment.fragment_id,
        v_node_id,
        v_clean_text,
        now(),
        now() + interval '1 day'
    )
    ON CONFLICT (retrieval_id, fragment_id, node_id)
    DO UPDATE SET
        recalled_text = EXCLUDED.recalled_text,
        created_at = now();

    -- Retire assignment to 'recalled' so it frees one capacity slot on this node
    UPDATE public.fragment_assignments
    SET status = 'recalled',
        last_verified_at = now()
    WHERE id = p_assignment_id;

    -- Delete any temporary delivery transport row
    DELETE FROM public.fragment_delivery_inbox
    WHERE assignment_id = p_assignment_id;

    RETURN jsonb_build_object(
        'success', true,
        'retrieval_id', p_retrieval_id,
        'assignment_id', p_assignment_id,
        'fragment_id', v_assignment.fragment_id,
        'message', 'Fragment recall accepted and node slot freed'
    );
END;
$$;

-- 8. Secure RPC: get_retrieval_status_and_fragments
CREATE OR REPLACE FUNCTION public.get_retrieval_status_and_fragments(p_retrieval_id UUID)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_user_id UUID;
    v_retrieval RECORD;
    v_responses JSONB;
    v_recalled_count INT;
BEGIN
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    -- Validate caller owns this retrieval
    SELECT r.id, r.memory_id, r.status, m.fragment_count
    INTO v_retrieval
    FROM public.memory_retrievals r
    JOIN public.memories m ON r.memory_id = m.id
    WHERE r.id = p_retrieval_id AND r.requester_user_id = v_user_id;

    IF v_retrieval.id IS NULL THEN
        RAISE EXCEPTION 'Retrieval not found or not owned by user';
    END IF;

    -- Fetch recall responses
    SELECT
        COALESCE(jsonb_agg(jsonb_build_object(
            'fragment_id', frr.fragment_id,
            'sequence_number', mf.sequence_number,
            'recalled_text', frr.recalled_text,
            'created_at', frr.created_at
        ) ORDER BY mf.sequence_number ASC), '[]'::JSONB),
        count(DISTINCT frr.fragment_id)::INT
    INTO v_responses, v_recalled_count
    FROM public.fragment_recall_responses frr
    JOIN public.memory_fragments mf ON frr.fragment_id = mf.id
    WHERE frr.retrieval_id = p_retrieval_id;

    RETURN jsonb_build_object(
        'retrieval_id', p_retrieval_id,
        'memory_id', v_retrieval.memory_id,
        'status', v_retrieval.status,
        'total_fragments', v_retrieval.fragment_count,
        'recalled_fragments_count', v_recalled_count,
        'responses', v_responses
    );
END;
$$;

-- 9. Secure RPC: complete_memory_retrieval
CREATE OR REPLACE FUNCTION public.complete_memory_retrieval(p_retrieval_id UUID)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_user_id UUID;
    v_retrieval RECORD;
    v_purged_count INT;
BEGIN
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    SELECT r.id, r.memory_id
    INTO v_retrieval
    FROM public.memory_retrievals r
    WHERE r.id = p_retrieval_id AND r.requester_user_id = v_user_id;

    IF v_retrieval.id IS NULL THEN
        RAISE EXCEPTION 'Retrieval not found or not owned by user';
    END IF;

    -- Mark completed
    UPDATE public.memory_retrievals
    SET status = 'completed',
        completed_at = now()
    WHERE id = p_retrieval_id;

    -- PURGE temporary transport plaintext responses immediately
    DELETE FROM public.fragment_recall_responses
    WHERE retrieval_id = p_retrieval_id;
    GET DIAGNOSTICS v_purged_count = ROW_COUNT;

    RETURN jsonb_build_object(
        'success', true,
        'retrieval_id', p_retrieval_id,
        'status', 'completed',
        'purged_transport_rows', v_purged_count
    );
END;
$$;

-- 10. Privileges and Ownership
REVOKE ALL ON FUNCTION public.get_user_stored_memories() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_user_stored_memories() TO authenticated;

REVOKE ALL ON FUNCTION public.initiate_memory_retrieval(UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.initiate_memory_retrieval(UUID) TO authenticated;

REVOKE ALL ON FUNCTION public.get_pending_recalls_for_node() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_pending_recalls_for_node() TO authenticated;

REVOKE ALL ON FUNCTION public.submit_fragment_recall(UUID, UUID, TEXT) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.submit_fragment_recall(UUID, UUID, TEXT) TO authenticated;

REVOKE ALL ON FUNCTION public.get_retrieval_status_and_fragments(UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_retrieval_status_and_fragments(UUID) TO authenticated;

REVOKE ALL ON FUNCTION public.complete_memory_retrieval(UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.complete_memory_retrieval(UUID) TO authenticated;

ALTER FUNCTION public.get_user_stored_memories() OWNER TO postgres;
ALTER FUNCTION public.initiate_memory_retrieval(UUID) OWNER TO postgres;
ALTER FUNCTION public.get_pending_recalls_for_node() OWNER TO postgres;
ALTER FUNCTION public.submit_fragment_recall(UUID, UUID, TEXT) OWNER TO postgres;
ALTER FUNCTION public.get_retrieval_status_and_fragments(UUID) OWNER TO postgres;
ALTER FUNCTION public.complete_memory_retrieval(UUID) OWNER TO postgres;
