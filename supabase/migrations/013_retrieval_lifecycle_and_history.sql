-- Migration: 013_retrieval_lifecycle_and_history.sql
-- Description:
--   1. Update public.memories status check constraint to include 'recovered'.
--   2. Update get_user_stored_memories to filter out completed/recovered memories
--      and provide live retrieval status ('ready' vs 'under_recovery').
--   3. Update initiate_memory_retrieval to reject duplicate recovery requests
--      for memories that have already completed recovery.
--   4. Update complete_memory_retrieval to transition parent memory status to 'recovered',
--      retire any remaining fragment assignments to 'recalled', and purge ephemeral plaintext.

-- 1. Ensure public.memories status check constraint allows 'recovered'
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT conname 
        FROM pg_constraint 
        WHERE conrelid = 'public.memories'::regclass 
          AND contype = 'c' 
          AND pg_get_constraintdef(oid) LIKE '%status%'
    ) LOOP
        EXECUTE 'ALTER TABLE public.memories DROP CONSTRAINT IF EXISTS ' || quote_ident(r.conname);
    END LOOP;
END $$;

ALTER TABLE public.memories ADD CONSTRAINT memories_status_check 
    CHECK (status IN ('active', 'expired', 'deleted', 'recovered'));

-- 2. Update get_user_stored_memories to return only active, unrecovered memories
--    with current retrieval lifecycle state ('ready' vs 'under_recovery')
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

    -- Return ONLY unrecovered, active memories
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', m.id,
        'created_at', m.created_at,
        'expires_at', m.expires_at,
        'fragment_count', m.fragment_count,
        'packet_count', m.fragment_count,
        'memorized_count', (
            SELECT count(DISTINCT fa.id)::INT
            FROM public.fragment_assignments fa
            WHERE fa.memory_id = m.id
              AND fa.status IN ('memorized', 'fulfilled')
        ),
        'retrieval_status', CASE
            WHEN EXISTS (
                SELECT 1 FROM public.memory_retrievals r
                WHERE r.memory_id = m.id AND r.status = 'open'
            ) THEN 'under_recovery'
            ELSE 'ready'
        END,
        'active_retrieval_id', (
            SELECT r.id FROM public.memory_retrievals r
            WHERE r.memory_id = m.id AND r.status = 'open'
            ORDER BY r.created_at DESC
            LIMIT 1
        ),
        'status', m.status
    ) ORDER BY m.created_at DESC), '[]'::JSONB)
    INTO v_result
    FROM public.memories m
    WHERE m.owner_user_id = v_user_id 
      AND m.status = 'active'
      AND NOT EXISTS (
          SELECT 1 FROM public.memory_retrievals mr
          WHERE mr.memory_id = m.id AND mr.status = 'completed'
      );

    RETURN v_result;
END;
$$;

-- 3. Update initiate_memory_retrieval with strict server-side duplicate recovery prevention
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
    v_existing_open_id UUID;
BEGIN
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    -- Validate ownership
    SELECT m.id, m.fragment_count, m.status
    INTO v_memory
    FROM public.memories m
    WHERE m.id = p_memory_id AND m.owner_user_id = v_user_id;

    IF v_memory.id IS NULL THEN
        RAISE EXCEPTION 'Memory not found or not owned by user';
    END IF;

    -- REJECTION 1: If memory has already been recovered
    IF v_memory.status = 'recovered' THEN
        RAISE EXCEPTION 'Memory has already been recovered and cannot be recovered again.';
    END IF;

    -- REJECTION 2: If any completed retrieval exists for this memory
    IF EXISTS (
        SELECT 1 FROM public.memory_retrievals r
        WHERE r.memory_id = p_memory_id AND r.status = 'completed'
    ) THEN
        RAISE EXCEPTION 'Memory has already completed recovery and cannot be recovered again.';
    END IF;

    IF v_memory.status != 'active' THEN
        RAISE EXCEPTION 'Memory is not active (status: %)', v_memory.status;
    END IF;

    -- Check if an open retrieval already exists
    SELECT r.id INTO v_existing_open_id
    FROM public.memory_retrievals r
    WHERE r.memory_id = p_memory_id AND r.status = 'open'
    ORDER BY r.created_at DESC
    LIMIT 1;

    IF v_existing_open_id IS NOT NULL THEN
        v_retrieval_id := v_existing_open_id;
    ELSE
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
        'holding_nodes_count', v_assigned_node_count,
        'is_resumed', (v_existing_open_id IS NOT NULL)
    );
END;
$$;

-- 4. Update complete_memory_retrieval to finalize lifecycle
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

    SELECT r.id, r.memory_id, r.status
    INTO v_retrieval
    FROM public.memory_retrievals r
    WHERE r.id = p_retrieval_id AND r.requester_user_id = v_user_id;

    IF v_retrieval.id IS NULL THEN
        RAISE EXCEPTION 'Retrieval not found or not owned by user';
    END IF;

    -- 1. Mark retrieval completed
    UPDATE public.memory_retrievals
    SET status = 'completed',
        completed_at = now()
    WHERE id = p_retrieval_id;

    -- 2. Mark memory status as 'recovered' so it never appears as active/recoverable again
    UPDATE public.memories
    SET status = 'recovered'
    WHERE id = v_retrieval.memory_id;

    -- 3. Retire any remaining active fragment assignments for this memory to 'recalled'
    UPDATE public.fragment_assignments
    SET status = 'recalled',
        last_verified_at = now()
    WHERE memory_id = v_retrieval.memory_id
      AND status IN ('assigned', 'fulfilled', 'memorized');

    -- 4. Delete temporary delivery transport records for this memory
    DELETE FROM public.fragment_delivery_inbox
    WHERE memory_id = v_retrieval.memory_id;

    -- 5. PURGE temporary transport plaintext responses immediately from DB
    DELETE FROM public.fragment_recall_responses
    WHERE retrieval_id = p_retrieval_id;
    GET DIAGNOSTICS v_purged_count = ROW_COUNT;

    RETURN jsonb_build_object(
        'success', true,
        'retrieval_id', p_retrieval_id,
        'memory_id', v_retrieval.memory_id,
        'status', 'completed',
        'purged_transport_rows', v_purged_count
    );
END;
$$;

-- 5. Privileges and ownership
ALTER FUNCTION public.get_user_stored_memories() OWNER TO postgres;
ALTER FUNCTION public.initiate_memory_retrieval(UUID) OWNER TO postgres;
ALTER FUNCTION public.complete_memory_retrieval(UUID) OWNER TO postgres;

GRANT EXECUTE ON FUNCTION public.get_user_stored_memories() TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.initiate_memory_retrieval(UUID) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.complete_memory_retrieval(UUID) TO authenticated, service_role;
