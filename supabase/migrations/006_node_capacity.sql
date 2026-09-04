-- Migration: 006_node_capacity.sql
-- Description: Strict 5-fragment capacity enforcement per human node:
--   1. Update fragment_assignments status check constraint to include 'recalled'
--   2. Add STABLE helper get_node_active_hosted_count(node_id)
--   3. Update assign_pending_fragments_for_available_nodes() to enforce capacity < 5 with FOR UPDATE lock
--   4. Update store_memory_with_fragments() to exclude full nodes from candidate selection
--   5. Add secure RPC retire_recalled_fragment_assignment() to retire an assignment upon successful retrieval/recall
--   6. Add secure RPC get_node_capacity_telemetry() for real-time station capacity reporting
--
-- Invariants:
--   - Maximum 5 active hosted fragments per node (status IN ('assigned', 'fulfilled', 'memorized')).
--   - Atomically locks candidate node row to prevent race conditions during concurrent assignments.
--   - Preserves existing over-capacity test nodes (does NOT delete existing data).
--   - Does NOT delete original memory_fragments metadata (owner retains reconstruction data).
--   - Safe to apply on databases where migrations 001-005 are already applied.

-- 1. Extend fragment_assignments status check constraint to support 'recalled'
ALTER TABLE public.fragment_assignments
    DROP CONSTRAINT IF EXISTS fragment_assignments_status_check;

ALTER TABLE public.fragment_assignments
    ADD CONSTRAINT fragment_assignments_status_check
    CHECK (status IN ('pending', 'assigned', 'fulfilled', 'memorized', 'recalled', 'unavailable'));

-- 2. Helper function to count active hosted fragments for a node
CREATE OR REPLACE FUNCTION public.get_node_active_hosted_count(p_node_id UUID)
RETURNS INT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public, extensions
AS $$
    SELECT count(*)::INT
    FROM public.fragment_assignments fa
    JOIN public.memory_fragments mf ON fa.fragment_id = mf.id
    JOIN public.memories m ON mf.memory_id = m.id
    WHERE fa.node_id = p_node_id
      AND fa.status IN ('assigned', 'fulfilled', 'memorized')
      AND m.status = 'active'
      AND m.expires_at > now()
      AND mf.expires_at > now();
$$;

-- 3. Secure RPC: assign_pending_fragments_for_available_nodes with strict capacity check (< 5)
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
    -- Authorization check
    v_caller_id := auth.uid();
    IF v_caller_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    -- Cleanup expired transport buffer rows and expired memories
    DELETE FROM public.fragment_delivery_inbox AS fdi WHERE fdi.expires_at <= now();
    UPDATE public.memories AS m SET status = 'expired' WHERE m.expires_at <= now() AND m.status = 'active';

    -- Iterate over pending assignments for active, unexpired memories with valid transport payload
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
        -- - Must strictly have LESS THAN 5 active hosted fragments
        -- - Row lock (FOR UPDATE) prevents race conditions across concurrent transactions
        SELECT n.id INTO v_chosen_node_id
        FROM public.nodes n
        WHERE n.status = 'online'
          AND n.user_id != r_pending.owner_user_id
          AND NOT EXISTS (
              SELECT 1 FROM public.fragment_assignments fa2
              WHERE fa2.fragment_id = r_pending.fragment_id
                AND fa2.node_id = n.id
          )
          AND (
              SELECT count(*)
              FROM public.fragment_assignments fa_cnt
              JOIN public.memory_fragments mf_cnt ON fa_cnt.fragment_id = mf_cnt.id
              JOIN public.memories m_cnt ON mf_cnt.memory_id = m_cnt.id
              WHERE fa_cnt.node_id = n.id
                AND fa_cnt.status IN ('assigned', 'fulfilled', 'memorized')
                AND m_cnt.status = 'active'
                AND m_cnt.expires_at > now()
                AND mf_cnt.expires_at > now()
          ) < 5
        ORDER BY n.reliability DESC, n.last_seen DESC
        LIMIT 1
        FOR UPDATE;

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

-- 4. Secure RPC: store_memory_with_fragments with strict node capacity filtering (< 5)
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
    v_active_count INT;
    v_memory_id UUID;
    v_expires_at TIMESTAMPTZ;
    v_frag_count INT;
    v_frag_idx INT;
    v_frag_text TEXT;
    v_frag_size INT;
    v_frag_hash TEXT;
    v_fragment_id UUID;
    v_candidate_node_ids UUID[];
    v_num_candidates INT;
    v_rep INT;
    v_target_node_id UUID;
    v_assignment_id UUID;
    v_assigned_replicas INT := 0;
    v_pending_replicas INT := 0;
    v_assigned_nodes_list TEXT[] := ARRAY[]::TEXT[];
    v_node_current_load INT;
BEGIN
    -- 1. Authorization check
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    -- 2. Input validation
    IF p_fragments IS NULL OR array_length(p_fragments, 1) IS NULL OR array_length(p_fragments, 1) = 0 THEN
        RAISE EXCEPTION 'Invalid fragments payload: Fragments array cannot be empty.';
    END IF;

    v_frag_count := array_length(p_fragments, 1);
    IF v_frag_count > 100 THEN
        RAISE EXCEPTION 'Memory limit exceeded: Maximum 100 fragments per memory allowed.';
    END IF;

    FOR v_frag_idx IN 1..v_frag_count LOOP
        IF p_fragments[v_frag_idx] IS NULL OR length(trim(p_fragments[v_frag_idx])) = 0 THEN
            RAISE EXCEPTION 'Invalid fragment at index %: Fragment text cannot be empty.', v_frag_idx;
        END IF;
    END LOOP;

    -- 3. Enforce maximum 5 active memories limit per user
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

    -- 6. Discover eligible online peer nodes with capacity < 5 (strictly excluding owner's node)
    SELECT array_agg(c.id) INTO v_candidate_node_ids
    FROM (
        SELECT n.id FROM public.nodes n
        WHERE n.user_id != v_user_id
          AND n.status = 'online'
          AND (
              SELECT count(*)
              FROM public.fragment_assignments fa_cnt
              JOIN public.memory_fragments mf_cnt ON fa_cnt.fragment_id = mf_cnt.id
              JOIN public.memories m_cnt ON mf_cnt.memory_id = m_cnt.id
              WHERE fa_cnt.node_id = n.id
                AND fa_cnt.status IN ('assigned', 'fulfilled', 'memorized')
                AND m_cnt.status = 'active'
                AND m_cnt.expires_at > now()
                AND mf_cnt.expires_at > now()
          ) < 5
        ORDER BY n.reliability DESC, n.last_seen DESC
        LIMIT 20
        FOR UPDATE OF n
    ) c;

    IF v_candidate_node_ids IS NOT NULL THEN
        v_num_candidates := array_length(v_candidate_node_ids, 1);
    ELSE
        v_num_candidates := 0;
    END IF;

    -- 7. Store fragment metadata and handle replica assignments with capacity enforcement
    FOR v_frag_idx IN 1..v_frag_count LOOP
        v_frag_text := p_fragments[v_frag_idx];
        v_frag_size := octet_length(v_frag_text);
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
            v_target_node_id := NULL;

            IF v_num_candidates > 0 AND v_rep <= v_num_candidates THEN
                v_target_node_id := v_candidate_node_ids[((v_frag_idx - 1) + (v_rep - 1)) % v_num_candidates + 1];

                -- Verify target node still has capacity < 5
                SELECT count(*) INTO v_node_current_load
                FROM public.fragment_assignments fa_chk
                JOIN public.memory_fragments mf_chk ON fa_chk.fragment_id = mf_chk.id
                JOIN public.memories m_chk ON mf_chk.memory_id = m_chk.id
                WHERE fa_chk.node_id = v_target_node_id
                  AND fa_chk.status IN ('assigned', 'fulfilled', 'memorized')
                  AND m_chk.status = 'active'
                  AND m_chk.expires_at > now()
                  AND mf_chk.expires_at > now();

                IF v_node_current_load >= 5 THEN
                    v_target_node_id := NULL;
                END IF;
            END IF;

            IF v_target_node_id IS NOT NULL THEN
                -- Assign distinct online peer node
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

                -- Insert into temporary transport delivery inbox (7-day lifetime)
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
                -- No peer node with available capacity (< 5): assignment is pending
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

-- 5. Secure RPC: retire_recalled_fragment_assignment
-- When a human node successfully provides a fragment during retrieval:
--   1. Recalled fragment is accepted by the retrieval system.
--   2. The assignment status is retired to 'recalled', freeing 1 slot from the 5-slot capacity.
--   3. Original memory_fragments metadata remains intact for owner reconstruction.
CREATE OR REPLACE FUNCTION public.retire_recalled_fragment_assignment(
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

    -- 2. Verify assignment belongs to caller node
    SELECT fa.id, fa.status, fa.fragment_id, fa.node_id
    INTO v_assignment
    FROM public.fragment_assignments AS fa
    WHERE fa.id = p_assignment_id AND fa.node_id = v_node_id;

    IF v_assignment.id IS NULL THEN
        RAISE EXCEPTION 'Assignment does not exist or does not belong to caller node';
    END IF;

    -- 3. Idempotent check: if already recalled, return true
    IF v_assignment.status = 'recalled' THEN
        RETURN TRUE;
    END IF;

    -- 4. Transition assignment to 'recalled' (frees this node's capacity slot)
    UPDATE public.fragment_assignments AS fa
    SET status = 'recalled',
        last_verified_at = now()
    WHERE fa.id = p_assignment_id;

    -- 5. Delete temporary transport delivery record if any remains
    DELETE FROM public.fragment_delivery_inbox AS fdi
    WHERE fdi.assignment_id = p_assignment_id;

    -- Note: memory_fragments record is NOT deleted; owner retains metadata
    RETURN TRUE;
END;
$$;

-- 6. Secure RPC: get_node_capacity_telemetry
CREATE OR REPLACE FUNCTION public.get_node_capacity_telemetry()
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_user_id UUID;
    v_node_id UUID;
    v_active_count INT := 0;
BEGIN
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RETURN jsonb_build_object('error', 'Not authenticated');
    END IF;

    SELECT n.id INTO v_node_id FROM public.nodes AS n WHERE n.user_id = v_user_id LIMIT 1;
    IF v_node_id IS NULL THEN
        RETURN jsonb_build_object('error', 'Node not found');
    END IF;

    SELECT count(*)::INT INTO v_active_count
    FROM public.fragment_assignments fa
    JOIN public.memory_fragments mf ON fa.fragment_id = mf.id
    JOIN public.memories m ON mf.memory_id = m.id
    WHERE fa.node_id = v_node_id
      AND fa.status IN ('assigned', 'fulfilled', 'memorized')
      AND m.status = 'active'
      AND m.expires_at > now()
      AND mf.expires_at > now();

    RETURN jsonb_build_object(
        'node_id', v_node_id,
        'active_hosted_count', v_active_count,
        'max_capacity', 5,
        'is_full', (v_active_count >= 5),
        'available_slots', GREATEST(0, 5 - v_active_count)
    );
END;
$$;

-- 7. Privileges and Ownership Configuration
REVOKE ALL ON FUNCTION public.get_node_active_hosted_count(UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_node_active_hosted_count(UUID) TO authenticated;

REVOKE ALL ON FUNCTION public.assign_pending_fragments_for_available_nodes() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.assign_pending_fragments_for_available_nodes() TO authenticated;

REVOKE ALL ON FUNCTION public.store_memory_with_fragments(TEXT[]) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.store_memory_with_fragments(TEXT[]) TO authenticated;

REVOKE ALL ON FUNCTION public.retire_recalled_fragment_assignment(UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.retire_recalled_fragment_assignment(UUID) TO authenticated;

REVOKE ALL ON FUNCTION public.get_node_capacity_telemetry() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_node_capacity_telemetry() TO authenticated;

ALTER FUNCTION public.get_node_active_hosted_count(UUID) OWNER TO postgres;
ALTER FUNCTION public.assign_pending_fragments_for_available_nodes() OWNER TO postgres;
ALTER FUNCTION public.store_memory_with_fragments(TEXT[]) OWNER TO postgres;
ALTER FUNCTION public.retire_recalled_fragment_assignment(UUID) OWNER TO postgres;
ALTER FUNCTION public.get_node_capacity_telemetry() OWNER TO postgres;
