-- Migration: 003_fix_delivery_rpc.sql
-- Description: Fix ambiguous column references (PostgrestException: 42702) in delivery RPCs.
--   Disambiguates column references (such as expires_at, assignment_id, fragment_id)
--   against PL/pgSQL variables and RETURNS TABLE output parameters using explicit table aliases.
--   Safe to run against an already-existing database; updates functions and permissions only.

-- 1. Secure RPC: store_memory_with_fragments
CREATE OR REPLACE FUNCTION public.store_memory_with_fragments(
    p_fragments TEXT[]
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_user_id UUID;
    v_frag_count INT;
    v_active_count INT;
    v_expires_at TIMESTAMPTZ;
    v_memory_id UUID;
    v_fragment_id UUID;
    v_assignment_id UUID;
    v_candidate_node_ids UUID[];
    v_num_candidates INT := 0;
    v_assigned_replicas INT := 0;
    v_pending_replicas INT := 0;
    v_assigned_nodes_list TEXT[] := ARRAY[]::TEXT[];
    v_target_node_id UUID;
    v_frag_idx INT;
    v_rep INT;
    v_frag_text TEXT;
    v_frag_size INT;
    v_frag_hash TEXT;
BEGIN
    -- 1. Authorization check
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    -- 2. Input validation: fragment array limits
    v_frag_count := array_length(p_fragments, 1);
    IF v_frag_count IS NULL OR v_frag_count <= 0 THEN
        RAISE EXCEPTION 'No fragments provided. At least 1 fragment is required.';
    END IF;

    IF v_frag_count > 100 THEN
        RAISE EXCEPTION 'Fragment count exceeds maximum allowed limit (maximum 100 fragments per memory).';
    END IF;

    -- Validate individual fragment sizes
    FOR v_frag_idx IN 1..v_frag_count LOOP
        v_frag_text := p_fragments[v_frag_idx];
        IF v_frag_text IS NULL OR length(trim(v_frag_text)) = 0 THEN
            RAISE EXCEPTION 'Fragment at position % is empty.', v_frag_idx;
        END IF;
        IF length(v_frag_text) > 2000 THEN
            RAISE EXCEPTION 'Fragment at position % exceeds maximum size limit of 2000 characters.', v_frag_idx;
        END IF;
    END LOOP;

    -- 3. Enforce maximum 5 active memories limit
    SELECT count(*) INTO v_active_count
    FROM public.memories m
    WHERE m.owner_user_id = v_user_id AND m.status = 'active';

    IF v_active_count >= 5 THEN
        RAISE EXCEPTION 'Active memory limit reached: Maximum 5 active memories allowed per user.';
    END IF;

    -- 4. Server-enforced 6-month lifetime (approx 180 days)
    v_expires_at := now() + interval '180 days';

    -- 5. Create memory metadata record (NO original plaintext stored)
    INSERT INTO public.memories (
        owner_user_id,
        created_at,
        expires_at,
        fragment_count,
        status
    )
    VALUES (
        v_user_id,
        now(),
        v_expires_at,
        v_frag_count,
        'active'
    )
    RETURNING id INTO v_memory_id;

    -- 6. Discover eligible online peer nodes (strictly excluding the owner's own node)
    SELECT array_agg(c.id) INTO v_candidate_node_ids
    FROM (
        SELECT n.id FROM public.nodes n
        WHERE n.user_id != v_user_id AND n.status = 'online'
        ORDER BY n.reliability DESC, n.last_seen DESC
        LIMIT 20
    ) c;

    IF v_candidate_node_ids IS NOT NULL THEN
        v_num_candidates := array_length(v_candidate_node_ids, 1);
    ELSE
        v_num_candidates := 0;
    END IF;

    -- 7. Store fragment metadata and handle replica assignments
    FOR v_frag_idx IN 1..v_frag_count LOOP
        v_frag_text := p_fragments[v_frag_idx];
        v_frag_size := octet_length(v_frag_text);
        -- SERVER-AUTHORITATIVE SHA-256 HASH COMPUTATION
        v_frag_hash := encode(extensions.digest(v_frag_text, 'sha256'), 'hex');

        -- Insert fragment METADATA ONLY into memory_fragments table
        INSERT INTO public.memory_fragments (
            memory_id,
            sequence_number,
            size_bytes,
            hash,
            created_at,
            expires_at
        )
        VALUES (
            v_memory_id,
            v_frag_idx,
            v_frag_size,
            v_frag_hash,
            now(),
            v_expires_at
        )
        RETURNING id INTO v_fragment_id;

        -- Create 3 replicas
        FOR v_rep IN 1..3 LOOP
            IF v_num_candidates > 0 AND v_rep <= v_num_candidates THEN
                -- Assign distinct online peer node (1 node per replica)
                v_target_node_id := v_candidate_node_ids[((v_frag_idx - 1) + (v_rep - 1)) % v_num_candidates + 1];

                INSERT INTO public.fragment_assignments (
                    fragment_id,
                    node_id,
                    replica_number,
                    assigned_at,
                    status
                )
                VALUES (
                    v_fragment_id,
                    v_target_node_id,
                    v_rep,
                    now(),
                    'assigned'
                )
                RETURNING id INTO v_assignment_id;

                -- Insert into temporary transport delivery inbox for the assigned node (7-day lifetime)
                INSERT INTO public.fragment_delivery_inbox (
                    assignment_id,
                    node_id,
                    fragment_id,
                    payload_text,
                    created_at,
                    expires_at
                )
                VALUES (
                    v_assignment_id,
                    v_target_node_id,
                    v_fragment_id,
                    v_frag_text,
                    now(),
                    now() + interval '7 days'
                );

                v_assigned_replicas := v_assigned_replicas + 1;
                IF NOT (v_target_node_id::TEXT = ANY(v_assigned_nodes_list)) THEN
                    v_assigned_nodes_list := array_append(v_assigned_nodes_list, v_target_node_id::TEXT);
                END IF;
            ELSE
                -- No peer node available: assignment is pending
                INSERT INTO public.fragment_assignments (
                    fragment_id,
                    node_id,
                    replica_number,
                    assigned_at,
                    status
                )
                VALUES (
                    v_fragment_id,
                    NULL,
                    v_rep,
                    now(),
                    'pending'
                )
                RETURNING id INTO v_assignment_id;

                -- Insert into temporary transport delivery inbox with node_id = NULL
                -- This preserves the temporary transport buffer (7 days) so pending rebalancing can assign it later
                INSERT INTO public.fragment_delivery_inbox (
                    assignment_id,
                    node_id,
                    fragment_id,
                    payload_text,
                    created_at,
                    expires_at
                )
                VALUES (
                    v_assignment_id,
                    NULL,
                    v_fragment_id,
                    v_frag_text,
                    now(),
                    now() + interval '7 days'
                );

                v_pending_replicas := v_pending_replicas + 1;
            END IF;
        END LOOP;
    END LOOP;

    RETURN jsonb_build_object(
        'memory_id', v_memory_id,
        'fragment_count', v_frag_count,
        'assigned_replicas', v_assigned_replicas,
        'fulfilled_replicas', 0,
        'pending_replicas', v_pending_replicas,
        'assigned_nodes', v_assigned_nodes_list
    );
END;
$$;

-- 2. Secure RPC: assign_pending_fragments_for_available_nodes
CREATE OR REPLACE FUNCTION public.assign_pending_fragments_for_available_nodes()
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_caller_id UUID;
    v_rebalanced_count INT := 0;
    r_pending RECORD;
    v_chosen_node_id UUID;
    v_inbox_payload TEXT;
BEGIN
    -- 1. Authorization check
    v_caller_id := auth.uid();
    IF v_caller_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    -- 2. Cleanup expired transport buffer rows and expired memories
    DELETE FROM public.fragment_delivery_inbox AS fdi WHERE fdi.expires_at <= now();
    UPDATE public.memories AS m SET status = 'expired' WHERE m.expires_at <= now() AND m.status = 'active';

    -- 3. Iterate over pending assignments for active, unexpired memories with valid transport payload
    FOR r_pending IN
        SELECT
            fa.id AS assignment_id,
            fa.fragment_id,
            fa.replica_number,
            mf.memory_id,
            m.owner_user_id
        FROM public.fragment_assignments fa
        JOIN public.memory_fragments mf ON fa.fragment_id = mf.id
        JOIN public.memories m ON mf.memory_id = m.id
        WHERE fa.status = 'pending'
          AND m.status = 'active'
          AND m.expires_at > now()
          AND mf.expires_at > now()
          -- Ensure unexpired transport payload exists in inbox
          AND EXISTS (
              SELECT 1 FROM public.fragment_delivery_inbox fdi
              WHERE (fdi.assignment_id = fa.id OR fdi.fragment_id = fa.fragment_id)
                AND fdi.expires_at > now()
          )
        ORDER BY fa.assigned_at ASC
    LOOP
        v_chosen_node_id := NULL;
        v_inbox_payload := NULL;

        -- Find an eligible online peer node:
        -- - Must be online
        -- - Cannot be the memory owner
        -- - Must NOT already host another replica of this fragment
        SELECT n.id INTO v_chosen_node_id
        FROM public.nodes n
        WHERE n.status = 'online'
          AND n.user_id != r_pending.owner_user_id
          AND NOT EXISTS (
              SELECT 1 FROM public.fragment_assignments fa2
              WHERE fa2.fragment_id = r_pending.fragment_id
                AND fa2.node_id = n.id
          )
        ORDER BY n.reliability DESC, n.last_seen DESC
        LIMIT 1;

        IF v_chosen_node_id IS NOT NULL THEN
            -- Update assignment to 'assigned'
            UPDATE public.fragment_assignments AS fa
            SET node_id = v_chosen_node_id,
                status = 'assigned',
                assigned_at = now()
            WHERE fa.id = r_pending.assignment_id;

            -- Update or create fragment_delivery_inbox transport row
            IF EXISTS (
                SELECT 1 FROM public.fragment_delivery_inbox AS fdi_chk
                WHERE fdi_chk.assignment_id = r_pending.assignment_id
            ) THEN
                UPDATE public.fragment_delivery_inbox AS fdi_up
                SET node_id = v_chosen_node_id,
                    expires_at = now() + interval '7 days'
                WHERE fdi_up.assignment_id = r_pending.assignment_id;
            ELSE
                -- Retrieve payload from any other inbox row for this fragment
                SELECT fdi_src.payload_text INTO v_inbox_payload
                FROM public.fragment_delivery_inbox AS fdi_src
                WHERE fdi_src.fragment_id = r_pending.fragment_id
                  AND fdi_src.expires_at > now()
                LIMIT 1;

                IF v_inbox_payload IS NOT NULL THEN
                    INSERT INTO public.fragment_delivery_inbox (
                        assignment_id,
                        node_id,
                        fragment_id,
                        payload_text,
                        created_at,
                        expires_at
                    )
                    VALUES (
                        r_pending.assignment_id,
                        v_chosen_node_id,
                        r_pending.fragment_id,
                        v_inbox_payload,
                        now(),
                        now() + interval '7 days'
                    );
                END IF;
            END IF;

            v_rebalanced_count := v_rebalanced_count + 1;
        END IF;
    END LOOP;

    RETURN jsonb_build_object(
        'rebalanced_count', v_rebalanced_count
    );
END;
$$;

-- 3. Secure RPC: get_pending_deliveries_for_node
CREATE OR REPLACE FUNCTION public.get_pending_deliveries_for_node()
RETURNS TABLE (
    inbox_id UUID,
    assignment_id UUID,
    fragment_id UUID,
    memory_id UUID,
    sequence_number INT,
    payload_text TEXT,
    size_bytes INT,
    hash TEXT,
    expires_at TIMESTAMPTZ,
    assigned_at TIMESTAMPTZ
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

    -- Cleanup any expired transport rows (qualified table column reference to avoid output variable ambiguity)
    DELETE FROM public.fragment_delivery_inbox AS fdi WHERE fdi.expires_at <= now();

    RETURN QUERY
    SELECT
        fdi.id AS inbox_id,
        fdi.assignment_id,
        fdi.fragment_id,
        mf.memory_id,
        mf.sequence_number,
        fdi.payload_text,
        mf.size_bytes,
        mf.hash,
        mf.expires_at,
        fa.assigned_at
    FROM public.fragment_delivery_inbox AS fdi
    JOIN public.fragment_assignments AS fa ON fdi.assignment_id = fa.id
    JOIN public.memory_fragments AS mf ON fdi.fragment_id = mf.id
    JOIN public.memories AS m ON mf.memory_id = m.id
    WHERE fdi.node_id = v_node_id
      AND fa.status = 'assigned'
      AND m.status = 'active'
      AND m.expires_at > now()
      AND mf.expires_at > now()
      AND fdi.expires_at > now()
    ORDER BY fdi.created_at ASC;
END;
$$;

-- 4. Secure RPC: confirm_fragment_receipt
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

    -- 6. Mark assignment as fulfilled
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

-- 5. Privileges and Superuser Configuration
REVOKE ALL ON FUNCTION public.store_memory_with_fragments(TEXT[]) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.store_memory_with_fragments(TEXT[]) TO authenticated;

REVOKE ALL ON FUNCTION public.assign_pending_fragments_for_available_nodes() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.assign_pending_fragments_for_available_nodes() TO authenticated;

REVOKE ALL ON FUNCTION public.get_pending_deliveries_for_node() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_pending_deliveries_for_node() TO authenticated;

REVOKE ALL ON FUNCTION public.confirm_fragment_receipt(UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.confirm_fragment_receipt(UUID) TO authenticated;

ALTER FUNCTION public.store_memory_with_fragments(TEXT[]) OWNER TO postgres;
ALTER FUNCTION public.assign_pending_fragments_for_available_nodes() OWNER TO postgres;
ALTER FUNCTION public.get_pending_deliveries_for_node() OWNER TO postgres;
ALTER FUNCTION public.confirm_fragment_receipt(UUID) OWNER TO postgres;
