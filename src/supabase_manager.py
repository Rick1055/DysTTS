from supabase import create_client, Client
import os
import io
from typing import Optional, Dict, Any, List, Union
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from datetime import datetime

class SupabaseManager:
    def __init__(self, supabase_url: Optional[str] = None,
                 supabase_key: Optional[str] = None,
                 use_service_role: bool = False):

        load_dotenv()

        self.url = (supabase_url or os.getenv("SUPABASE_URL")).strip().rstrip("/")

        if use_service_role:
            self.key = (supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or "").strip()
        else:
            self.key =(supabase_key or os.getenv("SUPABASE_KEY") or "").strip()

        if not self.url or not self.key:
            raise ValueError(
                "Missing Supabase credentials! Ensure SUPABASE_URL and SUPABASE_KEY are set in .env"
            )

        self.client: Client = create_client(self.url, self.key)
        self.bucket_name = "patient_audio"

        self.current_user = None
        self.current_profile = None

        print(f"[+] SupabaseManager initialized for: {self.url}")

    def sign_up(self, email:str,
                password: str,
                full_name: str,
                condition_type: str = "General Dysarthria",) -> Dict[str, Any]:

        email = email.strip().lower()
        if len(password) < 6:
            raise ValueError (
                "Password must be at least 6 characters long. "
            )

        response =self.client.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "full_name": full_name.strip(),
                        "condition_type": condition_type.strip(),
                    }
                },
            }
        )

        if not response.user:
            raise RuntimeError(f"Sign-up failed: {response}")

        self.current_user = response.user
        self.current_profile = self._fetch_or_create_profile(
            user_id = response.user.id,
            email = email,
            full_name = full_name,
            condition_type =condition_type,
        )

        print(f"[+] Patient registered successfully: {email} (ID: {self.current_profile.get('id')})")
        return {"user": self.current_user, "profile": self.current_profile}

    def sign_in(self, email: str, password: str) -> Dict[str, Any]:

        email = email.strip().lower()
        response = self.client.auth.sign_in_with_password(
            {"email": email, 
             "password": password}
        )

        if not response.user:
            raise RuntimeError (
                "Invalid Login Credentials"
            )

        self.current_user = response.user
        self.current_profile = self._fetch_or_create_profile(
            user_id = response.user.id,
            email = email,
        )

        print(f"[+] Patient Logged in: {email} (ID: {self.current_profile.get('id')})")
        return {"user": self.current_user, "profile": self.current_profile}

    def sign_out(self) -> None:

        try:
            self.client.auth.sign_out()
        except Exception:
            pass

        self.current_user = None
        self.current_profile = None
        print("[+] Patient Signed out.")

    def _fetch_or_create_profile(
            self,
            user_id: str,
            email: str,
            full_name: str = "Patient",
            condition_type: str = "General Dysarthria",
    ) -> Dict[str,Any]:

        res = (
            self.client.table("patient_profiles")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        if res.data and len(res.data) > 0:
            return res.data[0]

        insert_data = {
            "user_id": user_id,
            "email": email,
            "full_name": full_name,
            "condition_type": condition_type,
            "voice_profile": {},
            "phonetic_corrections": {},
        }
        fallback_res = self.client.table("patient_profiles").insert(insert_data).execute()
        return fallback_res.data[0] if fallback_res.data else insert_data

    def get_profile(self, refresh: bool = False) -> Optional[Dict[str, Any]]:

        if not self.current_user:
            return None

        if refresh or not self.current_profile:
            res = (
                self.client.table("patient_profiles")
                .select("*")
                .eq("user_id", self.current_user.id)
                .execute()
            )
            if res.data:
                self.current_profile = res.data[0]

        return self.current_profile

    def update_voice_profile(self, voice_palette: Dict[str, Any]) -> Dict[str, Any]:

        if not self.current_profile:
            raise RuntimeError(
                "No authenticated patient profile active."
            )
        
        patient_id = self.current_profile["id"]
        res = (
            self.client.table("patient_profiles")
            .update({"voice_profile": voice_palette})
            .eq("id", patient_id)
            .execute()
        )

        if res.data:
            self.current_profile["voice_profile"] = voice_palette
            print(f"[+] Voice profile updated for patient {patient_id}")
            return res.data[0]
        
        return self.current_profile

    def update_phonetic_corrections(self, corrections: Dict[str, str]) -> Dict[str, Any]:

        if not self.current_profile:
            raise RuntimeError (
                "No authenticated patient profile active"
            )

        patient_id =self.current_profile["id"]

        current = self.current_profile.get("phonetic_corrections") or {}
        current.update(corrections)

        res = (
            self.client.table("patient_profiles")
            .update({"phonetic_corrections": current})
            .eq("id", patient_id)
            .execute()
        )
        if res.data:
            self.current_profile["phonetic_corrections"] = current
            print(f"[+] Phonetic Corrections updated for patient {patient_id}")
            return res.data[0]
        return self.current_profile

    def upload_audio(self, audio_data: Union[np.ndarray, bytes, str],
                     remote_filename: str,
                     patient_id: Optional[str] = None,
                     folder: str = "raw",
                     sample_rate: int = 16000,
                     ) -> str:

        pid = patient_id or (self.current_profile["id"] if self.current_profile else "anonymous")
        remote_path = f"{pid}/{folder}/{remote_filename}"

        if isinstance(audio_data, str):
            if not os.path.exists(audio_data):
                raise FileNotFoundError (f"Audio file not found: {audio_data}")
            with open(audio_data, "rb") as f:
                file_bytes = f.read()
        elif isinstance(audio_data, bytes):
            file_bytes = audio_data
        elif isinstance(audio_data, (np.ndarray,list)):
            audio_arr = np.asarray(audio_data, dtype=np.float32)

            if audio_arr.ndim > 1:
                audio_arr = np.mean(audio_arr, axis = -1)

            audio_arr = np.nan_to_num(audio_arr)
            audio_arr = np.clip(audio_arr, -1.0, 1.0)

            buffer = io.BytesIO()
            sf.write(buffer, audio_arr, samplerate= sample_rate, format="WAV", subtype="PCM_16")
            buffer.seek(0)
            file_bytes = buffer.read()
        elif hasattr(audio_data, "detach"):
            audio_arr = audio_data.detach().cpu().numpy().squeeze()
            return self.upload_audio(audio_arr, remote_filename, pid, folder, sample_rate)
        else:
            raise TypeError (
                f"Unsupported audio type: {type(audio_data)}"
            )

        try:
            self.client.storage.from_(self.bucket_name).upload(
                path=remote_path,
                file=file_bytes,
                file_options = {"content-type": "audio/wav", "upsert": "true"},
            )
        except Exception as e:
            print(f"[!] Storage upload notice: {e}")

        public_url = self.client.storage.from_(self.bucket_name).get_public_url(remote_path)

        return public_url

    def start_therapy_session(self, patient_id: Optional[str] = None) -> str:

        pid = patient_id or (self.current_profile["id"] if self.current_profile else None)
        if not pid:
            raise RuntimeError ("Cannot start Session: No authenticated profile active.")

        session_data = {
            "patient_id": pid,
            "session_date": datetime.now().date().isoformat(),
            "duration_seconds": 0.0,
            "overall_score": 0.0,
            "utterances_count": 0,
            "accuracy_rate": 0.0,
        }

        res = self.client.table("therapy_sessions").insert(session_data).execute()
        if not res.data:
            raise RuntimeError("Failed to create therapy session.")

        session_id = res.data[0]["id"]
        print(f"[+] Started Therapy Session: {session_id} for patient {pid}")
        return session_id

    def log_utterance(self,
                      session_id: str,
                      transcribed_text: str,
                      corrected_text: Optional[str] = None,
                      raw_audio: Optional[Union[np.ndarray, bytes, str]] = None,
                      restored_audio: Optional[Union[np.ndarray, bytes, str]] = None,
                      confirmed: bool = True,
                      is_adapted: bool = False,
                      latency_ms: float = 0.0,
                      patient_id: Optional[str] = None,
                      ) -> Dict[str , Any]:

        pid = patient_id or (self.current_profile["id"] if self.current_profile else None)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        raw_url = None
        if raw_audio is not None:
            raw_url = self.upload_audio(
                audio_data = raw_audio,
                remote_filename=f"raw_{timestamp}.wav",
                patient_id = pid,
                folder="raw",
            )

        restored_url = None
        if restored_audio is not None:
            restored_url = self.upload_audio(
                audio_data=restored_audio,
                remote_filename=f"restored_{timestamp}.wav",
                patient_id=pid,
                folder="restored",
            )

        utterance_record = {
            "session_id": session_id,
            "patient_id": pid,
            "raw_audio_url": raw_url,
            "restored_audio_url": restored_url,
            "transcribed_text": transcribed_text,
            "corrected_text": corrected_text or transcribed_text,
            "confirmed": confirmed,
            "is_adapted": is_adapted,
            "latency_ms": round(float(latency_ms), 2)
        }

        res = self.client.table("session_utterances").insert(utterance_record).execute()
        return res.data[0] if res.data else utterance_record

    def log_biomarkers(self,
                       session_id: str,
                       biomarkers: Dict[str, Any],
                       patient_id: Optional[str] = None,
                       ) -> Dict[str, Any]:

        pid = patient_id or (self.current_profile["id"] if self.current_profile else None)
        record = {
            "session_id": session_id,
            "patient_id": pid,
            "articulation_velocity": round(float(biomarkers.get("articulation_velocity", 0.0)), 2),
            "speaking_rate": round(float(biomarkers.get("speaking_rate", 0.0)),2),
            "pause_ratio": round(float(biomarkers.get("pause_ratio", 0.0)),3),
            "formant_centralization_ratio": round(float(biomarkers.get("formant_centralization_ratio") or  biomarkers.get("fcr", 0.0)), 3),
            "vocal_jitter_pct": round(float(biomarkers.get("vocal_jitter_pct") or biomarkers.get("jitter_percent", 0.0)), 2),
            "vocal_shimmer_pct":  round(float(biomarkers.get("vocal_shimmer_pct") or biomarkers.get("shimmer_percent", 0.0)), 2),
            "mean_f0_hz": round(float(biomarkers.get("mean_f0_hz") or biomarkers.get("mean_f0", 0.0)),2),
            "attention_entropy": round(float(biomarkers.get("attention_entropy", 0.0)),4),
            "metrics": biomarkers,
        }

        res = self.client.table("biomarker_logs").insert(record).execute()
        return res.data[0] if res.data else record

    def finish_therapy_session(
            self,
            session_id: str,
            duration_seconds: float,
            overall_score: float,
            accuracy_rate: float,
            utterances_count: Optional[int] = None,
    ) -> Dict[str, Any]:

        update_data = {
            "duration_seconds": round(float(duration_seconds),2),
            "overall_score": round(float(overall_score), 2),
            "accuracy_rate": round(float(accuracy_rate), 2),
        }

        if utterances_count is not None:
            update_data["utterances_count"] = int(utterances_count)

        res = (
            self.client.table("therapy_sessions")
            .update(update_data)
            .eq("id", session_id)
            .execute()
        )
        print(f"[+] Finalized Session {session_id} - Score: {overall_score:.1f}% | Accuracy: {accuracy_rate:.1f}%")
        return res.data[0] if res.data else update_data
    
    def get_recovery_trajectory(
        self,
        patient_id: Optional[str] = None,
        limit_days: int = 30,
    ) -> List[Dict[str, Any]]:
       
        pid = patient_id or (self.current_profile["id"] if self.current_profile else None)
        if not pid:
            return []
        res = (
            self.client.table("patient_daily_progress")
            .select("*")
            .eq("patient_id", pid)
            .order("session_date", desc=False)
            .limit(limit_days)
            .execute()
        )

        return res.data or []