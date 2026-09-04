-- Migration: 005_fix_memorization_confirmation.sql
-- Description: Hotfix for confirm_fragment_receipt RPC:
--   1. Replaces public.confirm_fragment_receipt(UUID) with the corrected implementation
--      that marks fragment assignments with status 'memorized' and supports idempotent confirmation
--      (preventing errors during retries if the transport inbox record was already purged).
--   2. Grants EXECUTE to authenticated users, revokes PUBLIC access, and sets owner to postgres.
--
-- Safe to execute against databases with migrations 001-004 already applied.

CREATE OR REPLACE FUNCTION public.confirm_fragment_receipt(
    p_assignment_id UUID
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_user_id UUID;
    v_node_id UUID;
    v_assignment RECORD;
    v_inbox_exists BOOLEAN;
BEGIN
    -- 1. Authorization check
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    SELECT n.id INTO v_node_id FROM public.nodes AS n WHERE n.user_id = v_user_id LIMIT 1;
    IF v_node_id IS NULL THEN
        RAISE EXCEPTION 'Node not found for authenticated user';
    END IF;

    -- 2. Verify assignment belongs to caller node and fetch lifecycle status
    SELECT
        fa.id,
        fa.status,
        fa.node_id,
        mf.expires_at AS frag_expires_at,
        m.status AS mem_status,
        m.expires_at AS mem_expires_at
    INTO v_assignment
    FROM public.fragment_assignments AS fa
    JOIN public.memory_fragments AS mf ON fa.fragment_id = mf.id
    JOIN public.memories AS m ON mf.memory_id = m.id
    WHERE fa.id = p_assignment_id AND fa.node_id = v_node_id;

    IF v_assignment.id IS NULL THEN
        RAISE EXCEPTION 'Assignment does not exist or does not belong to caller node';
    END IF;

    -- 3. Verify assignment status is eligible ('assigned', or already 'memorized'/'fulfilled' in retry)
    IF v_assignment.status NOT IN ('assigned', 'memorized', 'fulfilled') THEN
        RAISE EXCEPTION 'Assignment status is %, expected assigned or memorized', v_assignment.status;
    END IF;

    -- 4. Verify memory and fragment have not expired
    IF v_assignment.frag_expires_at <= now() OR v_assignment.mem_expires_at <= now() OR v_assignment.mem_status != 'active' THEN
        RAISE EXCEPTION 'Assignment or memory has expired';
    END IF;

    -- 5. Verify temporary delivery inbox record exists and is unexpired
    SELECT EXISTS (
        SELECT 1 FROM public.fragment_delivery_inbox AS fdi
        WHERE fdi.assignment_id = p_assignment_id
          AND fdi.node_id = v_node_id
          AND fdi.expires_at > now()
    ) INTO v_inbox_exists;

    IF NOT v_inbox_exists THEN
        -- If already memorized or fulfilled and inbox row is gone, treat as idempotent success
        IF v_assignment.status IN ('memorized', 'fulfilled') THEN
            RETURN TRUE;
        END IF;
        RAISE EXCEPTION 'Temporary delivery inbox record does not exist or has expired';
    END IF;

    -- 6. Mark assignment as memorized (human node has memorized the fragment)
    UPDATE public.fragment_assignments AS fa
    SET status = 'memorized',
        last_verified_at = now()
    WHERE fa.id = p_assignment_id;

    -- 7. Delete temporary transport delivery record (zero permanent plaintext on server)
    DELETE FROM public.fragment_delivery_inbox AS fdi
    WHERE fdi.assignment_id = p_assignment_id;

    RETURN TRUE;
END;
$$;

-- Privileges & Ownership Configuration
REVOKE ALL ON FUNCTION public.confirm_fragment_receipt(UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.confirm_fragment_receipt(UUID) TO authenticated;
ALTER FUNCTION public.confirm_fragment_receipt(UUID) OWNER TO postgres;
