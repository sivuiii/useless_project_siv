-- Migration: 001_create_nodes.sql
-- Description: Create nodes table with constraints and RLS for Human Server node registration

CREATE TABLE IF NOT EXISTS public.nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'online' CHECK (status IN ('online', 'offline')),
    reliability FLOAT8 NOT NULL DEFAULT 1.0 CHECK (reliability >= 0.0 AND reliability <= 1.0),
    response_rate FLOAT8 NOT NULL DEFAULT 1.0 CHECK (response_rate >= 0.0 AND response_rate <= 1.0),
    avg_response_ms INT8 NOT NULL DEFAULT 0 CHECK (avg_response_ms >= 0),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for quick lookup by user_id
CREATE INDEX IF NOT EXISTS idx_nodes_user_id ON public.nodes(user_id);

-- Enable Row Level Security
ALTER TABLE public.nodes ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if any to prevent conflicts on re-runs
DROP POLICY IF EXISTS "Users can view their own node" ON public.nodes;
DROP POLICY IF EXISTS "Users can insert their own node" ON public.nodes;
DROP POLICY IF EXISTS "Users can update their own node" ON public.nodes;

-- Policy: Users can view their own node and discover online peer nodes
CREATE POLICY "Users can view their own node"
    ON public.nodes
    FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id OR status = 'online');

-- Policy: Users can only insert their own node
CREATE POLICY "Users can insert their own node"
    ON public.nodes
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

-- Policy: Users can only update their own node
CREATE POLICY "Users can update their own node"
    ON public.nodes
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);
