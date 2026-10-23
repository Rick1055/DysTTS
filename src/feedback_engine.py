import os
import json
from datetime import datetime
import numpy as np
import soundfile

class FeedbackEngine():
    def __init__(self, patient_id="patient__001", storage_dir = "data/patient_profiles"):
        self.patient_id = patient_id

        self.base_dir =os.path.join(storage_dir, self.patient_id)
        self.audio_dir = os.path.join(self.base_dir , "audio")

        self.history_file = os.path.join(self.base_dir, "history.jsonl")
        self.stats_file = os.path.join(self.base_dir, "stats.json")

        os.makedirs(self.audio_dir, exist_ok=True)

        self.stats = self._load_stats()
        print(f"[+] Loaded Profile for Patient: {self.patient_id}")
        print(f"    Total utterances logged: {self.stats['total_utterances']}")
        print(f"    Historical Accuracy    : {self.stats['accuracy_rate']:.2f}")

    def _load_stats(self):

        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, mode="r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            "patient_id": self.patient_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_utterances": 0,
            "accepted_first_try": 0,
            "corrected_by_user": 0,
            "accuracy_rate": 0.0,
            "pending_adaption_count": 0,
            "common_corrections": {}
        }

    def _save_stats(self):
        self.stats["last_updated"] = datetime.now().isoformat()
        if self.stats["total_utterances"] > 0:
            self.stats["accuracy_rate"] = (
                self.stats["accepted_first_try"]/self.stats["total_utterances"] * 100.0
            )

        with open(self.stats_file, mode="w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    def log_interaction(self, audio, hypothesis_text, confirmed, corrected_text=None, sample_rate=16000):
        self.stats["total_utterances"]+=1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"utt_{self.stats['total_utterances']:04d}_{timestamp}.wav"
        audio_filepath = os.path.join(self.audio_dir, audio_filename)

        if audio is not None and len(audio) > 0:
            soundfile.write(audio_filepath, audio, samplerate = sample_rate)
            duration_s = len(audio)/sample_rate
        else:
            duration_s = 0.0

        if confirmed == True:
            self.stats["accepted_first_try"]+=1
            final_text = hypothesis_text.strip()
            feedback_type = "accepted"
        else:
            self.stats["corrected_by_user"]+=1
            final_text = corrected_text.strip() if corrected_text else hypothesis_text.strip()
            feedback_type = "corrected"

            confusion_key = f"{hypothesis_text.strip().lower() } -> {final_text.lower()}"
            self.stats["common_corrections"][confusion_key] = self.stats["common_corrections"].get(confusion_key, 0)+1

        self.stats["pending_adaption_count"]+=1
        self._save_stats()

        log_entry = {
            "utterance_id":self.stats["total_utterances"],
            "timestamp": datetime.now().isoformat(),
            "audio_file": audio_filepath,
            "duration_s": round(duration_s,2),
            "hypothesis_text": hypothesis_text.strip(),
            "confirmed": confirmed,
            "corrected_text": corrected_text.strip() if corrected_text else None,
            "final_text": final_text,
            "feedback_type": feedback_type,
            "is_adapted": False,
        }

        with open(self.history_file , mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry)+"\n")

        return log_entry

    def get_adaption_queue(self):
        if not os.path.exists(self.history_file):
            return []

        pending_samples = []
        with open(self.history_file, mode="r", encoding="utf-8") as f:
            for line in f:
               if line.strip():
                entry = json.loads(line)
                if not entry.get("is_adapted", False):
                    pending_samples.append({
                        "utterance_id": entry["utterance_id"],
                        "audio_file": entry["audio_file"],
                        "target_text": entry["final_text"],
                        "feedback_type": entry["feedback_type"]
                })

        return pending_samples

    def mark_queue_adapted(self, adapted_utterance_ids):
        if not os.path.exists(self.history_file):
            return

        updated_entries=[]
        with open(self.history_file, mode="r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry["utterance_id"] in adapted_utterance_ids:
                        entry["is_adapted"] = True
                    updated_entries.append(entry)

        with open(self.history_file, mode="w", encoding="utf-8") as f:
            for entry in updated_entries:
                f.write(json.dumps(entry) + "\n")

        self.stats["pending_adaption_count"] = max(0, self.stats["pending_adaption_count"]-len(adapted_utterance_ids))
        self._save_stats()

if __name__ == "__main__":
    print("=" * 65)
    print("TESTING PATIENT FEEDBACK & CONTINUAL ADAPTATION ENGINE")
    print("=" * 65)

    # 1. Initialize for a test patient
    engine = FeedbackEngine(patient_id="test_patient_001")

    # 2. Simulate 1-second test audio waveform
    dummy_audio = np.random.uniform(-0.05, 0.05, 16000).astype(np.float32)

    # Simulation 1: Correct transcription confirmed (YES)
    print("\n[>] Test 1: Patient confirms correct transcription ('meat')")
    rec1 = engine.log_interaction(
        audio=dummy_audio,
        hypothesis_text="meat",
        confirmed=True
    )
    print(f"    AI Guessed : {rec1['hypothesis_text']}")
    print(f"    Confirmed? : {rec1['confirmed']}")
    print(f"    Final Text : {rec1['final_text']}")

    # Simulation 2: Slurred speech corrected (NO -> 'yard loss')
    print("\n[>] Test 2: Patient corrects misinterpretation ('yard less' -> 'yard loss')")
    rec2 = engine.log_interaction(
        audio=dummy_audio,
        hypothesis_text="a yard less",
        confirmed=False,
        corrected_text="a yard loss"
    )
    print(f"    AI Guessed : {rec2['hypothesis_text']}")
    print(f"    Confirmed? : {rec2['confirmed']}")
    print(f"    Corrected  : {rec2['final_text']}")

    # Check pending adaptation queue
    queue = engine.get_adaption_queue()
    print(f"\n[+] Adaptation Queue: {len(queue)} items waiting for model training:")
    for item in queue:
        print(f"    - Utterance #{item['utterance_id']}: Target='{item['target_text']}' ({item['feedback_type']})")

    # Simulate model learning and marking queue as adapted
    print("\n[>] Simulating background GPU micro-learning...")
    engine.mark_queue_adapted([item["utterance_id"] for item in queue])
    remaining = engine.get_adaption_queue()
    print(f"[+] After Training: {len(remaining)} items remaining in queue (expected: 0)")

    # Print final dashboard statistics
    print("\n" + "=" * 65)
    print("PATIENT DASHBOARD METRICS:")
    print(f"  Total Utterances  : {engine.stats['total_utterances']}")
    print(f"  First-Try Accuracy: {engine.stats['accuracy_rate']:.1f}%")
    print(f"  Confusion Mapping : {engine.stats['common_corrections']}")
    print("=" * 65)