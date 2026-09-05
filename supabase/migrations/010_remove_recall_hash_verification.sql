-- Migration: 010_remove_recall_hash_verification.sql
-- Description: Human Server Frozen MVP Correction:
--   1. Replace submit_fragment_recall to REMOVE server-side SHA-256 hash verification.
--      Humans are the actual storage; the server must not verify recall against a canonical hash.
--   2. Accept the human's manual recall text into the ephemeral transport buffer.
--   3. Retire the assignment to 'recalled' and immediately free the human's memory slot.
--   4. Delete temporary delivery transport records.
--   5. Maintain zero permanent plaintext on server.

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
    SELECT fa.id, fa.fragment_id, fa.status, mf.memory_id
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

    -- NO SERVER-SIDE SHA-256 HASH VERIFICATION:
    -- The human network itself is the storage. The server coordinates metadata
    -- and provides an ephemeral transport buffer, but does not arbitrate truth
    -- via canonical plaintext hashes.

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
        'message', 'Recall recorded successfully'
    );
END;
$$;

-- Security permissions
ALTER FUNCTION public.submit_fragment_recall(UUID, UUID, TEXT) OWNER TO postgres;
REVOKE ALL ON FUNCTION public.submit_fragment_recall(UUID, UUID, TEXT) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.submit_fragment_recall(UUID, UUID, TEXT) TO authenticated, service_role;
