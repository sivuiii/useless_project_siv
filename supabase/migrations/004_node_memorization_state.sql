-- Migration: 004_node_memorization_state.sql
-- Description: Human Server memorization architecture:
--   1. Update fragment_assignments status check constraint to include 'memorized'
--   2. Add secure RPC get_memorized_fragments_for_node: returns metadata ONLY (no plaintext)
--   3. Ensure confirm_fragment_receipt marks assignment as 'fulfilled' or 'memorized' and purges transport inbox
--   4. Grant privileges to authenticated users and set security definer ownership

-- 1. Update fragment_assignments status check constraint to support 'memorized'
ALTER TABLE public.fragment_assignments
    DROP CONSTRAINT IF EXISTS fragment_assignments_status_check;

ALTER TABLE public.fragment_assignments
    ADD CONSTRAINT fragment_assignments_status_check
    CHECK (status IN ('pending', 'assigned', 'fulfilled', 'memorized', 'unavailable'));

-- 2. Secure RPC: get_memorized_fragments_for_node
-- Returns metadata ONLY for all fragments memorized/fulfilled by the caller's node.
-- Never returns plaintext.
CREATE OR REPLACE FUNCTION public.get_memorized_fragments_for_node()
RETURNS TABLE (
    assignment_id UUID,
    fragment_id UUID,
    memory_id UUID,
    sequence_number INT,
    size_bytes INT,
    hash TEXT,
    expires_at TIMESTAMPTZ,
    assigned_at TIMESTAMPTZ,
    last_verified_at TIMESTAMPTZ,
    status TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_user_id UUID;
    v_node_id UUID;
BEGIN
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RETURN;
    END IF;

    SELECT n.id INTO v_node_id FROM public.nodes AS n WHERE n.user_id = v_user_id LIMIT 1;
    IF v_node_id IS NULL THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT
        fa.id AS assignment_id,
        fa.fragment_id,
        mf.memory_id,
        mf.sequence_number,
        mf.size_bytes,
        mf.hash,
        mf.expires_at,
        fa.assigned_at,
        fa.last_verified_at,
        fa.status
    FROM public.fragment_assignments AS fa
    JOIN public.memory_fragments AS mf ON fa.fragment_id = mf.id
    JOIN public.memories AS m ON mf.memory_id = m.id
    WHERE fa.node_id = v_node_id
      AND fa.status IN ('fulfilled', 'memorized')
      AND m.status = 'active'
      AND m.expires_at > now()
      AND mf.expires_at > now()
    ORDER BY mf.sequence_number ASC, fa.assigned_at DESC;
END;
$$;

-- 3. Update confirm_fragment_receipt to mark assignment as 'fulfilled' (memorized) and delete transport row
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

    -- 3. Verify assignment status is currently 'assigned'
    IF v_assignment.status != 'assigned' THEN
        RAISE EXCEPTION 'Assignment status is %, expected assigned', v_assignment.status;
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
        RAISE EXCEPTION 'Temporary delivery inbox record does not exist or has expired';
    END IF;

    -- 6. Mark assignment as fulfilled (human node has memorized the fragment)
    UPDATE public.fragment_assignments AS fa
    SET status = 'fulfilled',
        last_verified_at = now()
    WHERE fa.id = p_assignment_id;

    -- 7. Delete temporary transport delivery record (zero permanent plaintext on server)
    DELETE FROM public.fragment_delivery_inbox AS fdi
    WHERE fdi.assignment_id = p_assignment_id;

    RETURN TRUE;
END;
$$;

-- 4. Explicit RPC Privileges Configuration
REVOKE ALL ON FUNCTION public.get_memorized_fragments_for_node() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_memorized_fragments_for_node() TO authenticated;

REVOKE ALL ON FUNCTION public.confirm_fragment_receipt(UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.confirm_fragment_receipt(UUID) TO authenticated;

ALTER FUNCTION public.get_memorized_fragments_for_node() OWNER TO postgres;
ALTER FUNCTION public.confirm_fragment_receipt(UUID) OWNER TO postgres;
