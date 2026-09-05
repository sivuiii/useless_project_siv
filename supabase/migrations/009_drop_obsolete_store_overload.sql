-- Migration: 009_drop_obsolete_store_overload.sql
-- Description: Drop obsolete store_memory_with_fragments(TEXT[]) overload.
-- PostgREST cannot choose between store_memory_with_fragments(p_fragments => jsonb)
-- and store_memory_with_fragments(p_fragments => text[]) when called with RPC parameter 'p_fragments'.
-- The frozen MVP specification strictly uses store_memory_with_fragments(p_fragments => jsonb)
-- with client-generated canonical indices.

-- 1. Safely revoke permissions on obsolete overload
DO $$
BEGIN
    REVOKE ALL ON FUNCTION public.store_memory_with_fragments(TEXT[]) FROM PUBLIC;
    REVOKE ALL ON FUNCTION public.store_memory_with_fragments(TEXT[]) FROM authenticated;
    REVOKE ALL ON FUNCTION public.store_memory_with_fragments(TEXT[]) FROM service_role;
EXCEPTION WHEN OTHERS THEN
    NULL;
END;
$$;

-- 2. Drop obsolete TEXT[] overloads
DROP FUNCTION IF EXISTS public.store_memory_with_fragments(TEXT[]);
DROP FUNCTION IF EXISTS public.store_memory_with_fragments(TEXT[], TEXT);

-- 3. Confirm and re-grant permissions on the single canonical JSONB function
ALTER FUNCTION public.store_memory_with_fragments(JSONB) OWNER TO postgres;
REVOKE ALL ON FUNCTION public.store_memory_with_fragments(JSONB) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.store_memory_with_fragments(JSONB) TO authenticated, service_role;
