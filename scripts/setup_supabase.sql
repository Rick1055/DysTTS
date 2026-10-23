CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS public.patient_profiles(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    condition_type TEXT DEFAULT 'General Dysarthria',
    voice_profile JSONB DEFAULT '{}'::jsonb,
    phonetic_corrections JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.therapy_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES public.patient_profiles(id) ON DELETE CASCADE NOT NULL,
    session_date DATE DEFAULT CURRENT_DATE,
    duration_seconds NUMERIC(8,2) DEFAULT 0.0,
    overall_score NUMERIC(5,2) DEFAULT 0.0,
    utterances_count INTEGER DEFAULT 0,
    accuracy_rate NUMERIC(5,2) DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.biomarker_logs(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES public.therapy_sessions(id) ON DELETE CASCADE NOT NULL,
    patient_id UUID REFERENCES public.patient_profiles(id) ON DELETE CASCADE NOT NULL,
    articulation_velocity NUMERIC(5,2),
    speaking_rate NUMERIC(5,2),
    pause_ratio NUMERIC(5,3),
    formant_centralization_ratio NUMERIC(5,3),
    vocal_jitter_pct NUMERIC(5,2),
    vocal_shimmer_pct NUMERIC(5,2),
    mean_f0_hz NUMERIC(6,2),
    attention_entropy NUMERIC(5,4),
    metrics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.session_utterances(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES public.therapy_sessions(id) ON DELETE CASCADE NOT NULL,
    patient_id UUID REFERENCES public.patient_profiles(id) ON DELETE CASCADE NOT NULL,
    raw_audio_url TEXT,
    restored_audio_url TEXT,
    transcribed_text TEXT NOT NULL,
    corrected_text TEXT,
    confirmed BOOLEAN DEFAULT TRUE,
    is_adapted BOOLEAN DEFAULT FALSE,
    latency_ms NUMERIC(8,2) DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_therapy_sessions_patient_id ON
public.therapy_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_therapy_sessions_date ON
public.therapy_sessions(session_date);
CREATE INDEX IF NOT EXISTS idx_biomarker_logs_patient_id ON
public.biomarker_logs(patient_id);
CREATE INDEX IF NOT EXISTS idx_biomarker_logs_session_id ON
public.biomarker_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_session_utterances_patient_id ON
public.session_utterances(patient_id);
CREATE INDEX IF NOT EXISTS idx_session_utterances_session_id ON
public.session_utterances(session_id);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_patient_profiles_updated_at ON public.patient_profiles;
CREATE TRIGGER trigger_patient_profiles_updated_at
BEFORE UPDATE ON public.patient_profiles
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.patient_profiles (user_id, email, full_name, condition_type)
    VALUES(
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', 'Patient'),
        COALESCE(NEW.raw_user_meta_data->>'condition_type', 'General Dysarthria')
    )
    ON CONFLICT (user_id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
AFTER INSERT ON auth.users
FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();


INSERT INTO storage.buckets (id, name, public)
VALUES('patient_audio', 'patient_audio', true)
ON CONFLICT (id) DO UPDATE SET public = true;

ALTER TABLE public.patient_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.therapy_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.biomarker_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.session_utterances ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Patients view own profile" ON public.patient_profiles;
CREATE POLICY "Patients view own profile"
ON public.patient_profiles FOR ALL
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Patients access own sessions" ON public.therapy_sessions;
CREATE POLICY "Patients access own sessions"
ON public.therapy_sessions FOR ALL
USING (patient_id IN (SELECT id FROM public.patient_profiles WHERE user_id = auth.uid()))
WITH CHECK (patient_id IN(SELECT id FROM public.patient_profiles WHERE auth.uid() = user_id));

DROP POLICY IF EXISTS "Patients access own biomarkers" ON public.biomarker_logs;
CREATE POLICY "Patients access own biomarkers"
ON public.biomarker_logs FOR ALL
USING (patient_id IN (SELECT id FROM public.patient_profiles WHERE user_id = auth.uid()))
WITH CHECK (patient_id IN(SELECT id FROM public.patient_profiles WHERE auth.uid() = user_id));

DROP POLICY IF EXISTS "Patients access own utterances" ON public.session_utterances;
CREATE POLICY "Patients access own utterances"
ON public.session_utterances FOR ALL
USING (patient_id IN (SELECT id FROM public.patient_profiles WHERE user_id = auth.uid()))
WITH CHECK (patient_id IN(SELECT id FROM public.patient_profiles WHERE auth.uid() = user_id));

DROP POLICY IF EXISTS "Allow authenticated audio uploads" ON storage.objects;
CREATE POLICY "Allow authenticated audio uploads"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'patient_audio');

DROP POLICY IF EXISTS "Allow read access to audio" ON storage.objects;
CREATE POLICY "Allow read access to audio"
ON storage.objects FOR SELECT
TO authenticated, anon
USING (bucket_id = 'patient_audio');

DROP POLICY IF EXISTS "Allow authenticated audio updates" ON storage.objects;
CREATE POLICY "Allow authenticated audio updates"
ON storage.objects FOR UPDATE
TO authenticated
USING (bucket_id = 'patient_audio')
WITH CHECK (bucket_id = 'patient_audio');

DROP POLICY IF EXISTS "Allow authenticated audio deletes" ON storage.objects;
CREATE POLICY "Allow authenticated audio deletes"
ON storage.objects FOR DELETE
TO authenticated
USING (bucket_id = 'patient_audio');

CREATE OR REPLACE VIEW public.patient_daily_progress 
WITH (security_invoker = true) AS
SELECT 
    p.id AS patient_id,
    p.full_name,
    p.condition_type,
    s.session_date,
    COUNT(DISTINCT s.id) AS total_sessions_today,
    ROUND(AVG(s.accuracy_rate), 2) AS avg_accuracy,
    ROUND(AVG(b.articulation_velocity), 2) AS avg_articulation_velocity,
    ROUND(AVG(b.formant_centralization_ratio), 3) AS avg_fcr,
    ROUND(AVG(b.vocal_jitter_pct), 2) AS avg_jitter,
    ROUND(AVG(b.vocal_shimmer_pct), 2) AS avg_shimmer
FROM public.patient_profiles p
JOIN public.therapy_sessions s ON s.patient_id = p.id
LEFT JOIN public.biomarker_logs b ON b.session_id = s.id
GROUP BY p.id, p.full_name, p.condition_type, s.session_date
ORDER BY s.session_date DESC;

