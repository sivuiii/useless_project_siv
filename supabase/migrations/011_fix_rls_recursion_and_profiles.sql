-- Migration: 011_fix_rls_recursion_and_profiles.sql
-- Description:
--   1. Break infinite recursion in public.memories RLS policies by replacing circular
--      cross-table subqueries with STABLE SECURITY DEFINER helper functions.
--   2. Ensure public.profiles table exists with proper RLS policies allowing authenticated
--      users to insert/update their own profile (auth.uid() = id) and view profiles.
--   3. Establish on_auth_user_created trigger on auth.users for automated profile row creation.
--   4. Modernize RLS policies across memory_fragments, fragment_assignments,
--      memory_retrievals, and fragment_recall_responses to eliminate all policy recursion.

-- 1. Ensure public.profiles table exists
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    username TEXT,
    credits INT DEFAULT 0,
    reliability FLOAT8 DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS on profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Automated trigger for new user signups
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
BEGIN
    INSERT INTO public.profiles (id, username, credits, reliability)
    VALUES (
        NEW.id,
        COALESCE(
            NEW.raw_user_meta_data->>'username',
            split_part(NEW.email, '@', 1),
            'node_user'
        ),
        0,
        1.0
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Drop existing profiles policies
DROP POLICY IF EXISTS "Users can view profiles" ON public.profiles;
DROP POLICY IF EXISTS "Users can view their own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can insert their own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can update their own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can delete their own profile" ON public.profiles;
DROP POLICY IF EXISTS "Public profiles are viewable by everyone" ON public.profiles;

-- Recreate clean profiles policies
CREATE POLICY "Users can view profiles"
    ON public.profiles
    FOR SELECT
    TO authenticated
    USING (true);

CREATE POLICY "Users can insert their own profile"
    ON public.profiles
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
    ON public.profiles
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can delete their own profile"
    ON public.profiles
    FOR DELETE
    TO authenticated
    USING (auth.uid() = id);

-- 2. Non-recursive SECURITY DEFINER Authorization Helpers
-- These functions execute with elevated definer privileges and bypass RLS on subqueries,
-- preventing circular policy evaluation between memories <-> memory_fragments <-> fragment_assignments.

CREATE OR REPLACE FUNCTION public.is_memory_owner(p_memory_id UUID, p_user_id UUID)
RETURNS BOOLEAN
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, extensions
STABLE
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.memories
        WHERE id = p_memory_id
          AND owner_user_id = p_user_id
    );
$$;

CREATE OR REPLACE FUNCTION public.is_node_for_memory(p_memory_id UUID, p_user_id UUID)
RETURNS BOOLEAN
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, extensions
STABLE
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.fragment_assignments fa
        JOIN public.nodes n ON n.id = fa.node_id
        WHERE fa.memory_id = p_memory_id
          AND n.user_id = p_user_id
    );
$$;

CREATE OR REPLACE FUNCTION public.is_node_for_fragment(p_fragment_id UUID, p_user_id UUID)
RETURNS BOOLEAN
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, extensions
STABLE
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.fragment_assignments fa
        JOIN public.nodes n ON n.id = fa.node_id
        WHERE fa.fragment_id = p_fragment_id
          AND n.user_id = p_user_id
    );
$$;

CREATE OR REPLACE FUNCTION public.is_user_node(p_node_id UUID, p_user_id UUID)
RETURNS BOOLEAN
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, extensions
STABLE
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.nodes
        WHERE id = p_node_id
          AND user_id = p_user_id
    );
$$;

-- Security grants for helpers
ALTER FUNCTION public.is_memory_owner(UUID, UUID) OWNER TO postgres;
ALTER FUNCTION public.is_node_for_memory(UUID, UUID) OWNER TO postgres;
ALTER FUNCTION public.is_node_for_fragment(UUID, UUID) OWNER TO postgres;
ALTER FUNCTION public.is_user_node(UUID, UUID) OWNER TO postgres;

GRANT EXECUTE ON FUNCTION public.is_memory_owner(UUID, UUID) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.is_node_for_memory(UUID, UUID) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.is_node_for_fragment(UUID, UUID) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.is_user_node(UUID, UUID) TO authenticated, service_role;

-- 3. Reset and Recreate Memories RLS Policies
DROP POLICY IF EXISTS "Users can view their own memories" ON public.memories;
DROP POLICY IF EXISTS "Nodes can view active memories for assigned fragments" ON public.memories;
DROP POLICY IF EXISTS "Users can insert their own memories" ON public.memories;
DROP POLICY IF EXISTS "Users can update their own memories" ON public.memories;
DROP POLICY IF EXISTS "Users can delete their own memories" ON public.memories;

ALTER TABLE public.memories ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own memories"
    ON public.memories
    FOR SELECT
    TO authenticated
    USING (
        auth.uid() = owner_user_id
        OR public.is_node_for_memory(id, auth.uid())
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

CREATE POLICY "Users can delete their own memories"
    ON public.memories
    FOR DELETE
    TO authenticated
    USING (auth.uid() = owner_user_id);

-- 4. Reset and Recreate Memory Fragments RLS Policies
DROP POLICY IF EXISTS "Users can view metadata of fragments for their memories" ON public.memory_fragments;
DROP POLICY IF EXISTS "Nodes can view metadata of assigned fragments" ON public.memory_fragments;
DROP POLICY IF EXISTS "Users can insert metadata of fragments for their memories" ON public.memory_fragments;

ALTER TABLE public.memory_fragments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users and nodes can view fragment metadata"
    ON public.memory_fragments
    FOR SELECT
    TO authenticated
    USING (
        public.is_memory_owner(memory_id, auth.uid())
        OR public.is_node_for_fragment(id, auth.uid())
    );

CREATE POLICY "Users can insert fragments for their memories"
    ON public.memory_fragments
    FOR INSERT
    TO authenticated
    WITH CHECK (public.is_memory_owner(memory_id, auth.uid()));

CREATE POLICY "Users can update fragments for their memories"
    ON public.memory_fragments
    FOR UPDATE
    TO authenticated
    USING (public.is_memory_owner(memory_id, auth.uid()));

-- 5. Reset and Recreate Fragment Assignments RLS Policies
DROP POLICY IF EXISTS "Nodes can view assignments directed to them" ON public.fragment_assignments;
DROP POLICY IF EXISTS "Memory owners can view assignment status" ON public.fragment_assignments;
DROP POLICY IF EXISTS "Nodes can update their own assignments" ON public.fragment_assignments;

ALTER TABLE public.fragment_assignments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Nodes and owners can view fragment assignments"
    ON public.fragment_assignments
    FOR SELECT
    TO authenticated
    USING (
        public.is_user_node(node_id, auth.uid())
        OR public.is_memory_owner(memory_id, auth.uid())
        OR status = 'pending'
    );

CREATE POLICY "Nodes can update their own assignments"
    ON public.fragment_assignments
    FOR UPDATE
    TO authenticated
    USING (public.is_user_node(node_id, auth.uid()))
    WITH CHECK (public.is_user_node(node_id, auth.uid()));

-- 6. Reset and Recreate Delivery Inbox RLS Policies
DROP POLICY IF EXISTS "Nodes can read their own delivery inbox" ON public.fragment_delivery_inbox;
DROP POLICY IF EXISTS "Nodes can view pending delivery inbox" ON public.fragment_delivery_inbox;
DROP POLICY IF EXISTS "Nodes can update their own delivery inbox" ON public.fragment_delivery_inbox;
DROP POLICY IF EXISTS "Nodes can delete their own delivery inbox" ON public.fragment_delivery_inbox;

ALTER TABLE public.fragment_delivery_inbox ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Nodes can read their own delivery inbox"
    ON public.fragment_delivery_inbox
    FOR SELECT
    TO authenticated
    USING (public.is_user_node(node_id, auth.uid()));

CREATE POLICY "Nodes can update their own delivery inbox"
    ON public.fragment_delivery_inbox
    FOR UPDATE
    TO authenticated
    USING (public.is_user_node(node_id, auth.uid()));

CREATE POLICY "Nodes can delete their own delivery inbox"
    ON public.fragment_delivery_inbox
    FOR DELETE
    TO authenticated
    USING (public.is_user_node(node_id, auth.uid()));

-- 7. Reset and Recreate Memory Retrievals RLS Policies
DROP POLICY IF EXISTS "Users can view their own retrievals" ON public.memory_retrievals;
DROP POLICY IF EXISTS "Users can insert their own retrievals" ON public.memory_retrievals;
DROP POLICY IF EXISTS "Users can update their own retrievals" ON public.memory_retrievals;

ALTER TABLE public.memory_retrievals ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own retrievals"
    ON public.memory_retrievals
    FOR SELECT
    TO authenticated
    USING (
        auth.uid() = requester_user_id
        OR public.is_node_for_memory(memory_id, auth.uid())
    );

CREATE POLICY "Users can insert their own retrievals"
    ON public.memory_retrievals
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = requester_user_id);

CREATE POLICY "Users can update their own retrievals"
    ON public.memory_retrievals
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = requester_user_id);

-- 8. Reset and Recreate Fragment Recall Responses RLS Policies
DROP POLICY IF EXISTS "Nodes can view recall responses for their submissions" ON public.fragment_recall_responses;
DROP POLICY IF EXISTS "Memory owners can view recall responses for their retrievals" ON public.fragment_recall_responses;

ALTER TABLE public.fragment_recall_responses ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Nodes and owners can view recall responses"
    ON public.fragment_recall_responses
    FOR SELECT
    TO authenticated
    USING (
        public.is_user_node(node_id, auth.uid())
        OR EXISTS (
            SELECT 1 FROM public.memory_retrievals r
            WHERE r.id = retrieval_id
              AND r.requester_user_id = auth.uid()
        )
    );

CREATE POLICY "Nodes can insert recall responses"
    ON public.fragment_recall_responses
    FOR INSERT
    TO authenticated
    WITH CHECK (public.is_user_node(node_id, auth.uid()));

CREATE POLICY "Nodes can update recall responses"
    ON public.fragment_recall_responses
    FOR UPDATE
    TO authenticated
    USING (public.is_user_node(node_id, auth.uid()));

CREATE POLICY "Owners can delete recall responses upon completion"
    ON public.fragment_recall_responses
    FOR DELETE
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.memory_retrievals r
            WHERE r.id = retrieval_id
              AND r.requester_user_id = auth.uid()
        )
    );
