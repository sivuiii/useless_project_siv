-- Migration: 012_ensure_profiles_timestamp_defaults.sql
-- Description:
--   1. Ensure created_at and updated_at on public.profiles have DEFAULT now() and NOT NULL.
--   2. Backfill any existing NULL timestamp values.
--   3. Update handle_new_user() trigger to explicitly populate timestamps on insert.
--   4. Add automatic updated_at trigger for profile updates.
--   5. Idempotent: safe to run multiple times without data loss.

-- 1. Ensure columns exist with DEFAULT now()
ALTER TABLE public.profiles 
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now(),
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();

-- 2. Ensure defaults are explicitly set to now() in case columns were previously created without defaults
ALTER TABLE public.profiles 
    ALTER COLUMN created_at SET DEFAULT now(),
    ALTER COLUMN updated_at SET DEFAULT now();

-- 3. Backfill any rows that may have NULL timestamps
UPDATE public.profiles 
SET created_at = now() 
WHERE created_at IS NULL;

UPDATE public.profiles 
SET updated_at = now() 
WHERE updated_at IS NULL;

-- 4. Enforce NOT NULL constraint on timestamps
ALTER TABLE public.profiles 
    ALTER COLUMN created_at SET NOT NULL,
    ALTER COLUMN updated_at SET NOT NULL;

-- 5. Update auth trigger to explicitly include timestamps on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
BEGIN
    INSERT INTO public.profiles (
        id, 
        username, 
        credits, 
        reliability, 
        created_at, 
        updated_at
    )
    VALUES (
        NEW.id,
        COALESCE(
            NEW.raw_user_meta_data->>'username',
            split_part(NEW.email, '@', 1),
            'node_user'
        ),
        0,
        1.0,
        now(),
        now()
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$;

-- Ensure trigger exists on auth.users
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- 6. Automatically maintain updated_at on profile updates
CREATE OR REPLACE FUNCTION public.handle_profile_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_profile_updated ON public.profiles;
CREATE TRIGGER on_profile_updated
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_profile_updated_at();
