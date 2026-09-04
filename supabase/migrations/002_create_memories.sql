-- Migration: 002_create_memories.sql
-- Description: Human Server memory architecture MVP:
--   1. pgcrypto extension enabled for server-authoritative SHA-256 hashing
--   2. memories table (ownership, status, server-enforced 6-month expiry)
--   3. memory_fragments table (fragment METADATA only: size, server-computed hash, seq. NO permanent plaintext)
--   4. fragment_assignments table (replica assignment & fulfillment status)
--   5. fragment_delivery_inbox table (temporary 7-day transport buffer for node delivery, cleared upon receipt confirmation)
--   6. Secure RPCs:
--      - store_memory_with_fragments: Input-validated, server-authoritative hashing, 6-month expiry, atomic metadata creation
--      - assign_pending_fragments_for_available_nodes: Rebalances pending assignments to newly online peer nodes
--      - get_pending_deliveries_for_node: Fetches unexpired pending transport deliveries for caller node
--      - confirm_fragment_receipt: Strict caller verification, marks assignment fulfilled, purges temporary plaintext
--   7. Explicit RPC security: Revoke PUBLIC, grant strictly to authenticated

-- 0. Ensure pgcrypto extension is available in extensions schema
CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA extensions;

-- 1. Create memories table (ownership & lifecycle metadata only)
CREATE TABLE IF NOT EXISTS public.memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    fragment_count INT4 NOT NULL CHECK (fragment_count > 0),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'expired'))
);

CREATE INDEX IF NOT EXISTS idx_memories_owner ON public.memories(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_memories_status ON public.memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON public.memories(expires_at);

-- 2. Create memory_fragments table (METADATA ONLY - NO PLAINTEXT)
CREATE TABLE IF NOT EXISTS public.memory_fragments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES public.memories(id) ON DELETE CASCADE,
    sequence_number INT4 NOT NULL,
    size_bytes INT4 NOT NULL CHECK (size_bytes > 0),
    hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    UNIQUE (memory_id, sequence_number)
);

CREATE INDEX IF NOT EXISTS idx_memory_fragments_memory ON public.memory_fragments(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_fragments_expires_at ON public.memory_fragments(expires_at);

-- 3. Create fragment_assignments table
CREATE TABLE IF NOT EXISTS public.fragment_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fragment_id UUID NOT NULL REFERENCES public.memory_fragments(id) ON DELETE CASCADE,
    node_id UUID REFERENCES public.nodes(id) ON DELETE CASCADE,
    replica_number INT4 NOT NULL,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'assigned', 'fulfilled', 'unavailable')),
    last_verified_at TIMESTAMPTZ,
    UNIQUE (fragment_id, replica_number)
);

CREATE INDEX IF NOT EXISTS idx_assignments_fragment ON public.fragment_assignments(fragment_id);
CREATE INDEX IF NOT EXISTS idx_assignments_node ON public.fragment_assignments(node_id);
CREATE INDEX IF NOT EXISTS idx_assignments_status ON public.fragment_assignments(status);

-- 4. Create temporary fragment_delivery_inbox table (TRANSPORT BUFFER ONLY)
-- Plaintext resides here temporarily (expires after 7 days) until the assigned human node
-- downloads it locally and confirms receipt, at which point the row is immediately deleted.
-- node_id is nullable so pending replicas can hold their transport payload while awaiting peer nodes.
CREATE TABLE IF NOT EXISTS public.fragment_delivery_inbox (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assignment_id UUID NOT NULL UNIQUE REFERENCES public.fragment_assignments(id) ON DELETE CASCADE,
    node_id UUID REFERENCES public.nodes(id) ON DELETE CASCADE,
    fragment_id UUID NOT NULL REFERENCES public.memory_fragments(id) ON DELETE CASCADE,
    payload_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '7 days')
);

CREATE INDEX IF NOT EXISTS idx_inbox_node ON public.fragment_delivery_inbox(node_id);
CREATE INDEX IF NOT EXISTS idx_inbox_fragment ON public.fragment_delivery_inbox(fragment_id);
CREATE INDEX IF NOT EXISTS idx_inbox_expires ON public.fragment_delivery_inbox(expires_at);

-- 5. Enable Row Level Security (RLS) on all tables
ALTER TABLE public.memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.memory_fragments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fragment_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fragment_delivery_inbox ENABLE ROW LEVEL SECURITY;

-- Clean up existing policies
DROP POLICY IF EXISTS "Users can view their own memories" ON public.memories;
DROP POLICY IF EXISTS "Users can insert their own memories" ON public.memories;
DROP POLICY IF EXISTS "Users can update their own memories" ON public.memories;
DROP POLICY IF EXISTS "Nodes can view active memories for assigned fragments" ON public.memories;
DROP POLICY IF EXISTS "Nodes can view active memories with pending fragments" ON public.memories;

DROP POLICY IF EXISTS "Users can view metadata of fragments for their memories" ON public.memory_fragments;
DROP POLICY IF EXISTS "Nodes can view metadata of assigned fragments" ON public.memory_fragments;
DROP POLICY IF EXISTS "Nodes can view metadata of pending fragments" ON public.memory_fragments;

DROP POLICY IF EXISTS "Nodes can view assignments directed to them" ON public.fragment_assignments;
DROP POLICY IF EXISTS "Nodes can view pending assignments" ON public.fragment_assignments;
DROP POLICY IF EXISTS "Memory owners can view assignment status" ON public.fragment_assignments;
DROP POLICY IF EXISTS "Nodes can update their own assignments" ON public.fragment_assignments;

DROP POLICY IF EXISTS "Nodes can read their own delivery inbox" ON public.fragment_delivery_inbox;
DROP POLICY IF EXISTS "Nodes can view pending delivery inbox" ON public.fragment_delivery_inbox;
DROP POLICY IF EXISTS "Nodes can update their own delivery inbox" ON public.fragment_delivery_inbox;
DROP POLICY IF EXISTS "Nodes can delete their own delivery inbox" ON public.fragment_delivery_inbox;

-- Memories Policies
CREATE POLICY "Users can view their own memories"
    ON public.memories
    FOR SELECT
    TO authenticated
    USING (auth.uid() = owner_user_id);

CREATE POLICY "Nodes can view active memories for assigned fragments"
    ON public.memories
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.memory_fragments mf
            JOIN public.fragment_assignments fa ON fa.fragment_id = mf.id
            JOIN public.nodes n ON n.id = fa.node_id
            WHERE mf.memory_id = memories.id
              AND n.user_id = auth.uid()
        )
        OR EXISTS (
            SELECT 1 FROM public.memory_fragments mf
            JOIN public.fragment_assignments fa ON fa.fragment_id = mf.id
            WHERE mf.memory_id = memories.id
              AND fa.status = 'pending'
        )
    );

CREATE POLICY "Users can insert their own memories"
    ON public.memories
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = owner_user_id);

CREATE POLICY "Users can update their own memories"
    ON public.memories
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = owner_user_id)
    WITH CHECK (auth.uid() = owner_user_id);

-- Memory Fragments Policies (Metadata only)
CREATE POLICY "Users can view metadata of fragments for their memories"
    ON public.memory_fragments
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.memories m
            WHERE m.id = memory_fragments.memory_id
              AND m.owner_user_id = auth.uid()
        )
    );

CREATE POLICY "Nodes can view metadata of assigned fragments"
    ON public.memory_fragments
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.fragment_assignments fa
            JOIN public.nodes n ON n.id = fa.node_id
            WHERE fa.fragment_id = memory_fragments.id
              AND n.user_id = auth.uid()
        )
        OR EXISTS (
            SELECT 1 FROM public.fragment_assignments fa
            WHERE fa.fragment_id = memory_fragments.id
              AND fa.status = 'pending'
        )
    );

-- Fragment Assignments Policies
CREATE POLICY "Nodes can view assignments directed to them"
    ON public.fragment_assignments
    FOR SELECT
    TO authenticated
    USING (
        (
            node_id IS NOT NULL AND
            EXISTS (
                SELECT 1 FROM public.nodes n
                WHERE n.id = fragment_assignments.node_id
                  AND n.user_id = auth.uid()
            )
        )
        OR status = 'pending'
    );

CREATE POLICY "Memory owners can view assignment status"
    ON public.fragment_assignments
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.memory_fragments mf
            JOIN public.memories m ON mf.memory_id = m.id
            WHERE mf.id = fragment_assignments.fragment_id
              AND m.owner_user_id = auth.uid()
        )
    );

CREATE POLICY "Nodes can update their own assignments"
    ON public.fragment_assignments
    FOR UPDATE
    TO authenticated
    USING (
        (
            node_id IS NOT NULL AND
            EXISTS (
                SELECT 1 FROM public.nodes n
                WHERE n.id = fragment_assignments.node_id
                  AND n.user_id = auth.uid()
            )
        )
        OR status = 'pending'
    )
    WITH CHECK (
        (
            node_id IS NOT NULL AND
            EXISTS (
                SELECT 1 FROM public.nodes n
                WHERE n.id = fragment_assignments.node_id
                  AND n.user_id = auth.uid()
            )
        )
        OR status = 'pending'
    );

-- Fragment Delivery Inbox Policies:
-- Accessible by the assigned receiving node's authenticated user or when pending transport buffer
CREATE POLICY "Nodes can read their own delivery inbox"
    ON public.fragment_delivery_inbox
    FOR SELECT
    TO authenticated
    USING (
        (
            node_id IS NOT NULL AND
            EXISTS (
                SELECT 1 FROM public.nodes n
                WHERE n.id = fragment_delivery_inbox.node_id
                  AND n.user_id = auth.uid()
            )
        )
        OR node_id IS NULL
    );

CREATE POLICY "Nodes can update their own delivery inbox"
    ON public.fragment_delivery_inbox
    FOR UPDATE
    TO authenticated
    USING (
        (
            node_id IS NOT NULL AND
            EXISTS (
                SELECT 1 FROM public.nodes n
                WHERE n.id = fragment_delivery_inbox.node_id
                  AND n.user_id = auth.uid()
            )
        )
        OR node_id IS NULL
    )
    WITH CHECK (
        (
            node_id IS NOT NULL AND
            EXISTS (
                SELECT 1 FROM public.nodes n
                WHERE n.id = fragment_delivery_inbox.node_id
                  AND n.user_id = auth.uid()
            )
        )
        OR node_id IS NULL
    );

CREATE POLICY "Nodes can delete their own delivery inbox"
    ON public.fragment_delivery_inbox
    FOR DELETE
    TO authenticated
    USING (
        node_id IS NOT NULL AND
        EXISTS (
            SELECT 1 FROM public.nodes n
            WHERE n.id = fragment_delivery_inbox.node_id
              AND n.user_id = auth.uid()
        )
    );

-- 6. Secure RPC: store_memory_with_fragments
-- Features:
-- - Server-authoritative SHA-256 calculation via pgcrypto extensions.digest()
-- - Server-enforced 6-month expiration (now() + interval '180 days')
-- - Server-side input validation limits:
--     * Array cannot be empty
--     * Max 100 fragments per memory
--     * Individual fragment cannot be empty and cannot exceed 2000 characters
-- - Records memory & fragment metadata ONLY (no permanent plaintext in memory_fragments)
-- - Writes temporary 7-day transport payload to fragment_delivery_inbox
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

-- 7. Secure RPC: assign_pending_fragments_for_available_nodes
-- Rebalances pending fragment assignments to available online peer nodes:
-- - Discovers online peer nodes, strictly excluding the memory owner
-- - Never assigns two replicas of the same fragment to the same node
-- - Only processes active, unexpired memories (status = 'active' AND expires_at > now())
-- - Creates/updates temporary fragment_delivery_inbox rows for assigned replicas
-- - Updates assignment status from 'pending' to 'assigned'
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

-- 8. Secure RPC: get_pending_deliveries_for_node
-- Allows a node to download pending fragment deliveries from the transport inbox.
-- Rejects expired memories and cleans up expired transport rows.
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

-- 9. Secure RPC: confirm_fragment_receipt
-- Strictly verifies that:
-- - The caller is authenticated and owns the assignment node
-- - The assignment is currently 'assigned'
-- - The temporary delivery inbox row exists and is unexpired
-- - The memory and fragment have not expired
-- Once verified, marks assignment as 'fulfilled' and deletes the temporary transport row.
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

-- 10. Explicit RPC Privileges Configuration
-- Revoke execution from PUBLIC and grant strictly to authenticated users
REVOKE ALL ON FUNCTION public.store_memory_with_fragments(TEXT[]) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.store_memory_with_fragments(TEXT[]) TO authenticated;

REVOKE ALL ON FUNCTION public.assign_pending_fragments_for_available_nodes() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.assign_pending_fragments_for_available_nodes() TO authenticated;

REVOKE ALL ON FUNCTION public.get_pending_deliveries_for_node() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_pending_deliveries_for_node() TO authenticated;

REVOKE ALL ON FUNCTION public.confirm_fragment_receipt(UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.confirm_fragment_receipt(UUID) TO authenticated;

-- Ensure functions run with superuser privileges to bypass RLS when coordinating node deliveries
ALTER FUNCTION public.store_memory_with_fragments(TEXT[]) OWNER TO postgres;
ALTER FUNCTION public.assign_pending_fragments_for_available_nodes() OWNER TO postgres;
ALTER FUNCTION public.get_pending_deliveries_for_node() OWNER TO postgres;
ALTER FUNCTION public.confirm_fragment_receipt(UUID) OWNER TO postgres;
