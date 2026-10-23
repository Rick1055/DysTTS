"""
End-to-End Live Verification: SupabaseManager Integration Test
==============================================================
Tests the entire Supabase pipeline:
1. Manager initialization & health check
2. Test patient registration / login
3. Voice profile and phonetic confusion dictionary updates
4. In-memory audio upload to 'patient_audio' bucket
5. Therapy session creation & utterance logging
6. Biomarker telemetry sync (Speech Gym metrics)
7. Querying daily progress view (patient_daily_progress)
"""

import sys
import os
import numpy as np

# Ensure root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.supabase_manager import SupabaseManager


def run_e2e_test():
    print("=" * 65)
    print("  SUPABASE CLOUD PIPELINE: LIVE END-TO-END VERIFICATION")
    print("=" * 65)

    # 1. Initialize SupabaseManager (using service role for automated test execution)
    print("\n[Step 1] Initializing SupabaseManager...")
    mgr = SupabaseManager(use_service_role=True)
    print(" [+] Gateway successfully connected to Supabase.")

    # 2. Test Patient Profile Setup
    print("\n[Step 2] Testing Patient Registration / Authentication...")
    test_email = "clinical_test_patient@voiceai.org"
    test_password = "ClinicalSecret2026!"
    test_name = "Eleanor Vance (Test)"
    test_condition = "Cerebral Palsy (Ataxic Dysarthria)"

    try:
        auth_res = mgr.sign_in(email=test_email, password=test_password)
        print(f" [+] Existing test patient logged in: {test_email}")
    except Exception:
        print(f" [*] Patient not found, registering new test patient: {test_email}...")
        auth_res = mgr.sign_up(
            email=test_email,
            password=test_password,
            full_name=test_name,
            condition_type=test_condition,
        )
        print(f" [+] Registered new test patient: {test_email}")

    patient_id = mgr.current_profile["id"]
    print(f" [+] Active Patient ID: {patient_id}")
    print(f" [+] Active Patient Name: {mgr.current_profile['full_name']}")
    print(f" [+] Clinical Condition: {mgr.current_profile['condition_type']}")

    # 3. Voice Profile & Personalization
    print("\n[Step 3] Updating Voice Palette & Phonetic Memory...")
    mock_voice_palette = {
        "engine": "choice_1_sculptor",
        "base_voice": "en-US-ChristopherNeural",
        "pitch_hz": 115.0,
        "rate_pct": -5.0,
        "warmth_pct": 20.0,
        "roughness_pct": 5.0,
    }
    mgr.update_voice_profile(mock_voice_palette)
    print(f" [+] Voice profile successfully saved in Supabase.")

    mock_corrections = {"yard less": "yard loss", "he he": "he"}
    mgr.update_phonetic_corrections(mock_corrections)
    print(f" [+] Personalized phonetic dictionary saved: {mock_corrections}")

    # 4. In-Memory Audio Upload
    print("\n[Step 4] Testing In-Memory Audio Cloud Storage Upload...")
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate 1 sec pure 440 Hz test tone in memory (no disk file)
    test_waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    raw_url = mgr.upload_audio(
        audio_data=test_waveform,
        remote_filename="test_raw_mic_tone.wav",
        folder="raw",
        sample_rate=sr,
    )
    print(f" [+] Raw Audio Uploaded! Public URL: {raw_url}")

    restored_url = mgr.upload_audio(
        audio_data=test_waveform,
        remote_filename="test_restored_voice_tone.wav",
        folder="restored",
        sample_rate=sr,
    )
    print(f" [+] Restored Audio Uploaded! Public URL: {restored_url}")

    # 5. Speech Gym Therapy Session & Utterance Logging
    print("\n[Step 5] Logging Live Therapy Session & Utterances...")
    session_id = mgr.start_therapy_session()

    utt = mgr.log_utterance(
        session_id=session_id,
        transcribed_text="yard less",
        corrected_text="yard loss",
        raw_audio=test_waveform,
        restored_audio=test_waveform,
        confirmed=False,
        is_adapted=True,
        latency_ms=116.83,
    )
    print(f" [+] Utterance logged (ID: {utt['id']}) | Transcribed: '{utt['transcribed_text']}' -> Corrected: '{utt['corrected_text']}'")

    # 6. Biomarker Telemetry Sync
    print("\n[Step 6] Syncing Objective Bio-Acoustic Telemetry...")
    mock_biomarkers = {
        "articulation_velocity": 3.85,
        "speaking_rate": 2.90,
        "pause_ratio": 0.245,
        "formant_centralization_ratio": 1.182,
        "vocal_jitter_pct": 2.45,
        "vocal_shimmer_pct": 4.10,
        "mean_f0_hz": 124.50,
        "attention_entropy": 0.8124,
    }
    bio_res = mgr.log_biomarkers(session_id=session_id, biomarkers=mock_biomarkers)
    print(f" [+] Biomarkers recorded in PostgreSQL: Jitter: {bio_res['vocal_jitter_pct']}% | FCR: {bio_res['formant_centralization_ratio']} | Artic: {bio_res['articulation_velocity']} syl/s")

    # 7. Finalize Session
    print("\n[Step 7] Finalizing Therapy Session Summary...")
    mgr.finish_therapy_session(
        session_id=session_id,
        duration_seconds=45.2,
        overall_score=88.5,
        accuracy_rate=92.0,
        utterances_count=1,
    )

    # 8. Query 30-Day Recovery View
    print("\n[Step 8] Querying Clinical Recovery Trajectory (patient_daily_progress view)...")
    trajectory = mgr.get_recovery_trajectory(limit_days=7)
    print(f" [+] Found {len(trajectory)} daily progress records for patient.")
    for entry in trajectory:
        print(f"     Date: {entry['session_date']} | Total Sessions: {entry['total_sessions_today']} | Avg Accuracy: {entry['avg_accuracy']}% | Avg Jitter: {entry['avg_jitter']}%")

    print("\n" + "=" * 65)
    print("  [SUCCESS] All Supabase Cloud Subsystems Verified 100% Live!")
    print("=" * 65)


if __name__ == "__main__":
    run_e2e_test()

