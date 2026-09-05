-- Migration: 008_frozen_mvp_spec.sql
-- Description: Frozen Human Server MVP Specification Database Layer:
--   1. Schema updates for single-node-per-memory invariant (memory_id on fragment_assignments)
--   2. Strict partial unique index: A node can NEVER host more than 1 active fragment of the same memory
--   3. canonical_indices INT[] on memory_fragments to support deterministic reconstruction without plaintext
--   4. Update store_memory_with_fragments to enforce 1 fragment per different human node (no duplicates)
--   5. Update assign_pending_fragments_for_available_nodes to enforce 1 fragment per different human node
--   6. Update get_pending_deliveries_for_node to return sender's username (MEMORY FROM: [USERNAME])
--   7. Update get_pending_recalls_for_node to return sender's username (RECOVERY REQUEST: [USERNAME])
--   8. Update get_user_stored_memories to return packet_count and memorized_count
--   9. Add get_memory_packet_status to fetch live packet memorization counts without exposing recipients or plaintext

-- 1. Ensure extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA extensions;

-- 2. Add memory_id to public.fragment_assignments if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
          AND table_name = 'fragment_assignments' 
          AND column_name = 'memory_id'
    ) THEN
        ALTER TABLE public.fragment_assignments 
        ADD COLUMN memory_id UUID REFERENCES public.memories(id) ON DELETE CASCADE;
    END IF;
END $$;

-- Backfill memory_id on fragment_assignments from memory_fragments
UPDATE public.fragment_assignments fa
SET memory_id = mf.memory_id
FROM public.memory_fragments mf
WHERE fa.fragment_id = mf.id AND fa.memory_id IS NULL;

-- 3. Add canonical_indices to public.memory_fragments if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
          AND table_name = 'memory_fragments' 
          AND column_name = 'canonical_indices'
    ) THEN
        ALTER TABLE public.memory_fragments 
        ADD COLUMN canonical_indices INT[];
    END IF;
END $$;

-- 4. HARD MVP INVARIANT: A node can NEVER receive two fragments belonging to the same memory
-- Partial unique index ensures at the database engine level that no node has >1 active assignment per memory
CREATE UNIQUE INDEX IF NOT EXISTS uq_assignment_memory_node_active
ON public.fragment_assignments(memory_id, node_id)
WHERE status IN ('assigned', 'fulfilled', 'memorized');

CREATE INDEX IF NOT EXISTS idx_assignments_memory_id ON public.fragment_assignments(memory_id);

-- Drop existing functions to allow signature and return type updates safely
DROP FUNCTION IF EXISTS public.store_memory_with_fragments(JSONB);
DROP FUNCTION IF EXISTS public.store_memory_with_fragments(TEXT[]);
DROP FUNCTION IF EXISTS public.store_memory_with_fragments(TEXT[], TEXT);
DROP FUNCTION IF EXISTS public.assign_pending_fragments_for_available_nodes();
DROP FUNCTION IF EXISTS public.get_pending_deliveries_for_node();
DROP FUNCTION IF EXISTS public.get_pending_recalls_for_node();
DROP FUNCTION IF EXISTS public.get_retrieval_status_and_fragments(UUID);
DROP FUNCTION IF EXISTS public.get_user_stored_memories();
DROP FUNCTION IF EXISTS public.get_memory_packet_status(UUID);

-- 5. Secure RPC: store_memory_with_fragments (JSONB payload with text + canonical_indices)
-- Enforces:
-- - Client-side fragmented packets
-- - NO original plaintext stored permanently in the database
-- - Every segment goes to a DIFFERENT human node (zero duplicate nodes per memory)
-- - Max 5 active capacity per node enforced
-- - If insufficient nodes, marks excess fragments 'pending' (awaiting more human nodes)
-- - Never exposes recipient identities
CREATE OR REPLACE FUNCTION public.store_memory_with_fragments(
    p_fragments JSONB
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
    v_frag_item JSONB;
    v_frag_text TEXT;
    v_frag_indices INT[];
    v_frag_size INT;
    v_frag_hash TEXT;
    v_fragment_id UUID;
    v_candidate_node_ids UUID[];
    v_num_candidates INT;
    v_candidate_idx INT := 1;
    v_target_node_id UUID;
    v_assignment_id UUID;
    v_assigned_count INT := 0;
    v_pending_count INT := 0;
BEGIN
    -- Authorization check
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    -- Input validation
    IF p_fragments IS NULL OR jsonb_array_length(p_fragments) = 0 THEN
        RAISE EXCEPTION 'Invalid fragments payload: Fragments array cannot be empty.';
    END IF;

    v_frag_count := jsonb_array_length(p_fragments);
    IF v_frag_count > 100 THEN
        RAISE EXCEPTION 'Memory limit exceeded: Maximum 100 fragments per memory allowed.';
    END IF;

    -- Enforce maximum 5 active memories per user
    SELECT count(*) INTO v_active_count
    FROM public.memories m
    WHERE m.owner_user_id = v_user_id AND m.status = 'active';

    IF v_active_count >= 5 THEN
        RAISE EXCEPTION 'Active memory limit reached: Maximum 5 active memories allowed per user.';
    END IF;

    v_expires_at := now() + interval '180 days';

    -- Create memory metadata record (NO original plaintext stored)
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

    -- Discover eligible online peer nodes with capacity < 5, strictly excluding the owner
    -- Each candidate node will receive at most ONE fragment for this memory
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
        LIMIT 50
        FOR UPDATE OF n
    ) c;

    IF v_candidate_node_ids IS NOT NULL THEN
        v_num_candidates := array_length(v_candidate_node_ids, 1);
    ELSE
        v_num_candidates := 0;
    END IF;

    -- Iterate and insert fragments + assign to distinct nodes
    FOR v_frag_idx IN 1..v_frag_count LOOP
        v_frag_item := p_fragments->(v_frag_idx - 1);
        
        -- Support both {"text": "...", "indices": [...]} and plain string
        IF jsonb_typeof(v_frag_item) = 'object' THEN
            v_frag_text := v_frag_item->>'text';
            IF v_frag_item ? 'indices' AND jsonb_typeof(v_frag_item->'indices') = 'array' THEN
                SELECT array_agg(val::INT) INTO v_frag_indices
                FROM jsonb_array_elements_text(v_frag_item->'indices') AS val;
            ELSE
                v_frag_indices := ARRAY[v_frag_idx - 1];
            END IF;
        ELSE
            v_frag_text := v_frag_item #>> '{}';
            v_frag_indices := ARRAY[v_frag_idx - 1];
        END IF;

        IF v_frag_text IS NULL OR length(trim(v_frag_text)) = 0 THEN
            RAISE EXCEPTION 'Invalid fragment at index %: Fragment text cannot be empty.', v_frag_idx;
        END IF;

        v_frag_size := octet_length(v_frag_text);
        v_frag_hash := encode(extensions.digest(v_frag_text, 'sha256'), 'hex');

        -- Insert fragment METADATA ONLY into memory_fragments
        INSERT INTO public.memory_fragments (
            memory_id,
            sequence_number,
            size_bytes,
            hash,
            canonical_indices,
            created_at,
            expires_at
        )
        VALUES (
            v_memory_id,
            v_frag_idx,
            v_frag_size,
            v_frag_hash,
            v_frag_indices,
            now(),
            v_expires_at
        )
        RETURNING id INTO v_fragment_id;

        -- Hard MVP Requirement: EVERY SEGMENT MUST GO TO A DIFFERENT HUMAN
        v_target_node_id := NULL;
        IF v_candidate_idx <= v_num_candidates THEN
            v_target_node_id := v_candidate_node_ids[v_candidate_idx];
            v_candidate_idx := v_candidate_idx + 1;
        END IF;

        IF v_target_node_id IS NOT NULL THEN
            -- Assign to this distinct online peer node
            INSERT INTO public.fragment_assignments (
                memory_id,
                fragment_id,
                node_id,
                replica_number,
                assigned_at,
                status
            )
            VALUES (
                v_memory_id,
                v_fragment_id,
                v_target_node_id,
                1,
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

            v_assigned_count := v_assigned_count + 1;
        ELSE
            -- Insufficient distinct peer nodes: segment stays pending awaiting new human nodes
            INSERT INTO public.fragment_assignments (
                memory_id,
                fragment_id,
                node_id,
                replica_number,
                assigned_at,
                status
            )
            VALUES (
                v_memory_id,
                v_fragment_id,
                NULL,
                1,
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

            v_pending_count := v_pending_count + 1;
        END IF;
    END LOOP;

    -- Return sender packet status metadata ONLY (zero recipient identities)
    RETURN jsonb_build_object(
        'memory_id', v_memory_id,
        'packet_count', v_frag_count,
        'packets_assigned', v_assigned_count,
        'packets_pending', v_pending_count,
        'packets_memorized', 0,
        'status', CASE WHEN v_pending_count > 0 THEN 'awaiting_more_nodes' ELSE 'fully_assigned' END
    );
END;
$$;


-- 6. Secure RPC: assign_pending_fragments_for_available_nodes
-- Enforces:
-- - Hard MVP rule: A node can NEVER host more than one fragment of the same memory
-- - Enforces strict 5-fragment capacity cap
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
    v_caller_id := auth.uid();
    IF v_caller_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    DELETE FROM public.fragment_delivery_inbox AS fdi WHERE fdi.expires_at <= now();
    UPDATE public.memories AS m SET status = 'expired' WHERE m.expires_at <= now() AND m.status = 'active';

    FOR r_pending IN
        SELECT
            fa.id AS assignment_id,
            fa.fragment_id,
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

        -- Hard MVP Requirement:
        -- - Node must be online
        -- - Node cannot be the memory owner
        -- - Node must NEVER already host ANY fragment belonging to this memory
        -- - Node must have < 5 active memorized packets
        SELECT n.id INTO v_chosen_node_id
        FROM public.nodes n
        WHERE n.status = 'online'
          AND n.user_id != r_pending.owner_user_id
          AND NOT EXISTS (
              SELECT 1 FROM public.fragment_assignments fa2
              WHERE fa2.memory_id = r_pending.memory_id
                AND fa2.node_id = n.id
                AND fa2.status IN ('assigned', 'fulfilled', 'memorized')
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
        FOR UPDATE OF n;

        IF v_chosen_node_id IS NOT NULL THEN
            UPDATE public.fragment_assignments AS fa
            SET node_id = v_chosen_node_id,
                memory_id = r_pending.memory_id,
                status = 'assigned',
                assigned_at = now()
            WHERE fa.id = r_pending.assignment_id;

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
        'rebalanced_count', v_rebalanced_count,
        'status', 'success'
    );
END;
$$;

-- 7. Secure RPC: get_pending_deliveries_for_node
-- Returns sender's username so the recipient knows whose memory it is (MEMORY FROM: [USERNAME])
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
    assigned_at TIMESTAMPTZ,
    sender_username TEXT
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
        fa.assigned_at,
        COALESCE(p.username, split_part(u.email, '@', 1), 'node_user') AS sender_username
    FROM public.fragment_delivery_inbox AS fdi
    JOIN public.fragment_assignments AS fa ON fdi.assignment_id = fa.id
    JOIN public.memory_fragments AS mf ON fdi.fragment_id = mf.id
    JOIN public.memories AS m ON mf.memory_id = m.id
    LEFT JOIN public.profiles AS p ON m.owner_user_id = p.id
    LEFT JOIN auth.users AS u ON m.owner_user_id = u.id
    WHERE fdi.node_id = v_node_id
      AND fa.status = 'assigned'
      AND m.status = 'active'
      AND m.expires_at > now()
      AND mf.expires_at > now()
      AND fdi.expires_at > now()
    ORDER BY fdi.created_at ASC;
END;
$$;

-- -- 8. Secure RPC: get_pending_recalls_for_node
-- Returns sender's username so the human knows who is requesting memory recovery
-- (RECOVERY REQUEST: [USERNAME] IS REQUESTING MEMORY RECOVERY)
CREATE OR REPLACE FUNCTION public.get_pending_recalls_for_node()
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_caller_id UUID;
    v_node_id UUID;
    v_result JSONB;
BEGIN
    v_caller_id := auth.uid();
    IF v_caller_id IS NULL THEN
        RETURN '[]'::JSONB;
    END IF;

    SELECT id INTO v_node_id
    FROM public.nodes
    WHERE user_id = v_caller_id;

    IF v_node_id IS NULL THEN
        RETURN '[]'::JSONB;
    END IF;

    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'retrieval_id', r.id,
        'assignment_id', a.id,
        'fragment_id', f.id,
        'memory_id', m.id,
        'sequence_number', f.sequence_number,
        'size_bytes', f.size_bytes,
        'expected_hash', f.hash,
        'retrieval_created_at', r.created_at,
        'sender_username', COALESCE(p.username, split_part(u.email, '@', 1), 'node_user'),
        'canonical_indices', f.canonical_indices
    ) ORDER BY f.sequence_number ASC), '[]'::JSONB)
    INTO v_result
    FROM public.memory_retrievals r
    JOIN public.memories m ON m.id = r.memory_id
    JOIN public.memory_fragments f ON f.memory_id = m.id
    JOIN public.fragment_assignments a ON a.fragment_id = f.id
    LEFT JOIN public.profiles p ON m.owner_user_id = p.id
    LEFT JOIN auth.users u ON m.owner_user_id = u.id
    WHERE r.status = 'open'
      AND a.node_id = v_node_id
      AND a.status IN ('memorized', 'fulfilled')
      AND NOT EXISTS (
          SELECT 1 FROM public.fragment_recall_responses resp
          WHERE resp.retrieval_id = r.id
            AND resp.fragment_id = f.id
            AND resp.node_id = v_node_id
      );

    RETURN v_result;
END;
$$;

-- 9. Secure RPC: get_retrieval_status_and_fragments
-- Returns canonical_indices with each recalled fragment so the owner can accurately
-- assemble the message regardless of response arrival order
CREATE OR REPLACE FUNCTION public.get_retrieval_status_and_fragments(p_retrieval_id UUID)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_caller_id UUID;
    v_retrieval RECORD;
    v_total_fragments INTEGER;
    v_responses JSONB;
BEGIN
    v_caller_id := auth.uid();
    IF v_caller_id IS NULL THEN
        RAISE EXCEPTION 'Authentication required.';
    END IF;

    SELECT r.*, m.id AS mem_id INTO v_retrieval
    FROM public.memory_retrievals r
    JOIN public.memories m ON m.id = r.memory_id
    WHERE r.id = p_retrieval_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Retrieval request not found.';
    END IF;

    IF v_retrieval.requester_user_id <> v_caller_id THEN
        RAISE EXCEPTION 'Access denied.';
    END IF;

    SELECT COUNT(id)::INTEGER INTO v_total_fragments
    FROM public.memory_fragments
    WHERE memory_id = v_retrieval.memory_id;

    SELECT COALESCE(
        jsonb_agg(
            jsonb_build_object(
                'fragment_id', resp.fragment_id,
                'sequence_number', f.sequence_number,
                'recalled_text', resp.recalled_text,
                'canonical_indices', f.canonical_indices,
                'created_at', resp.created_at
            ) ORDER BY resp.created_at ASC
        ),
        '[]'::jsonb
    ) INTO v_responses
    FROM public.fragment_recall_responses resp
    JOIN public.memory_fragments f ON f.id = resp.fragment_id
    WHERE resp.retrieval_id = p_retrieval_id;

    RETURN jsonb_build_object(
        'retrieval_id', v_retrieval.id,
        'memory_id', v_retrieval.memory_id,
        'status', v_retrieval.status,
        'total_fragments', v_total_fragments,
        'recalled_fragments_count', jsonb_array_length(v_responses),
        'responses', v_responses
    );
END;
$$;

-- 10. Secure RPC: get_user_stored_memories
-- Returns packet_count and memorized_count (MEM-XXXX, 6 PACKETS, 4 / 6 MEMORIZED)
CREATE OR REPLACE FUNCTION public.get_user_stored_memories()
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_caller_id UUID;
    v_result JSONB;
BEGIN
    v_caller_id := auth.uid();
    IF v_caller_id IS NULL THEN
        RAISE EXCEPTION 'Authentication required.';
    END IF;

    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', m.id,
        'created_at', m.created_at,
        'expires_at', m.expires_at,
        'packet_count', m.fragment_count,
        'memorized_count', (
            SELECT COUNT(DISTINCT fa.fragment_id)::INTEGER
            FROM public.fragment_assignments fa
            JOIN public.memory_fragments mf ON fa.fragment_id = mf.id
            WHERE mf.memory_id = m.id
              AND fa.status IN ('memorized', 'fulfilled', 'recalled')
        ),
        'status', m.status
    ) ORDER BY m.created_at DESC), '[]'::JSONB)
    INTO v_result
    FROM public.memories m
    WHERE m.owner_user_id = v_caller_id
      AND m.status = 'active'
      AND m.expires_at > now();

    RETURN v_result;
END;
$$;

-- 11. Secure RPC: get_memory_packet_status
-- Returns live memorized packet counts for sender after dispatch (no plaintext, no recipients)
CREATE OR REPLACE FUNCTION public.get_memory_packet_status(p_memory_id UUID)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_caller_id UUID;
    v_memory RECORD;
    v_assigned_count INT;
    v_pending_count INT;
    v_memorized_count INT;
BEGIN
    v_caller_id := auth.uid();
    IF v_caller_id IS NULL THEN
        RAISE EXCEPTION 'Authentication required.';
    END IF;

    SELECT * INTO v_memory
    FROM public.memories
    WHERE id = p_memory_id AND owner_user_id = v_caller_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Memory not found or access denied.';
    END IF;

    SELECT 
        COUNT(*) FILTER (WHERE fa.status = 'assigned')::INT,
        COUNT(*) FILTER (WHERE fa.status = 'pending')::INT,
        COUNT(*) FILTER (WHERE fa.status IN ('memorized', 'fulfilled', 'recalled'))::INT
    INTO v_assigned_count, v_pending_count, v_memorized_count
    FROM public.fragment_assignments fa
    JOIN public.memory_fragments mf ON fa.fragment_id = mf.id
    WHERE mf.memory_id = p_memory_id;

    RETURN jsonb_build_object(
        'memory_id', p_memory_id,
        'packet_count', v_memory.fragment_count,
        'packets_assigned', v_assigned_count,
        'packets_pending', v_pending_count,
        'packets_memorized', v_memorized_count,
        'status', CASE WHEN v_pending_count > 0 THEN 'awaiting_more_nodes' ELSE 'distributed' END
    );
END;
$$;

-- 12. Security Grants
ALTER FUNCTION public.store_memory_with_fragments(JSONB) OWNER TO postgres;
ALTER FUNCTION public.assign_pending_fragments_for_available_nodes() OWNER TO postgres;
ALTER FUNCTION public.get_pending_deliveries_for_node() OWNER TO postgres;
ALTER FUNCTION public.get_pending_recalls_for_node() OWNER TO postgres;
ALTER FUNCTION public.get_retrieval_status_and_fragments(UUID) OWNER TO postgres;
ALTER FUNCTION public.get_user_stored_memories() OWNER TO postgres;
ALTER FUNCTION public.get_memory_packet_status(UUID) OWNER TO postgres;

REVOKE ALL ON FUNCTION public.store_memory_with_fragments(JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.assign_pending_fragments_for_available_nodes() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_pending_deliveries_for_node() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_pending_recalls_for_node() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_retrieval_status_and_fragments(UUID) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_user_stored_memories() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_memory_packet_status(UUID) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION public.store_memory_with_fragments(JSONB) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.assign_pending_fragments_for_available_nodes() TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.get_pending_deliveries_for_node() TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.get_pending_recalls_for_node() TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.get_retrieval_status_and_fragments(UUID) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.get_user_stored_memories() TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.get_memory_packet_status(UUID) TO authenticated, service_role;
