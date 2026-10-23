import re
import time

try:
    from .feedback_engine import FeedbackEngine
except ImportError:
    from feedback_engine import FeedbackEngine


class IntentRepair:
    """
    Contextual Intent, Dysfluency, and Stutter Repair Engine.
    
    1. Eliminates involuntary neuromotor syllable repetitions (e.g., 'he he' -> 'he').
    2. Applies patient-specific learned phonetic corrections from FeedbackEngine.
    3. Normalizes capitalization and punctuation.
    """

    def __init__(self, feedback_engine=None):
        self.feedback_engine = feedback_engine
        if self.feedback_engine:
            self.corrections = self.feedback_engine.stats.get("common_corrections", {})
        else:
            self.corrections = {}

    def clean_dysfluencies(self, text: str) -> str:
        """
        Removes consecutive repeated words caused by vocal spasms or stuttering.
        e.g., 'he he nailed it' -> 'he nailed it'
        """
        cleaned = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def repair(self, text: str):
        """
        Applies multi-tier dysfluency cleaning and patient memory repair.
        """
        start_time = time.time()
        original_text = text

        # Tier 1: Clean stutters and duplicate words
        text = self.clean_dysfluencies(text)

        # Tier 2: Apply patient-specific memory from FeedbackEngine
        if self.feedback_engine:
            self.corrections = self.feedback_engine.stats.get("common_corrections", {})
            for key in self.corrections:
                if "->" in key:
                    mistake, true_word = [part.strip() for part in key.split("->")]
                    if mistake.lower() in text.lower():
                        text = re.sub(rf'\b{re.escape(mistake)}\b', true_word, text, flags=re.IGNORECASE)

        # Tier 3: Capitalize first letter and ensure ending punctuation
        if text:
            text = text[0].upper() + text[1:]

        if text and text[-1] not in ".!?":
            text = text + "."

        latency_ms = (time.time() - start_time) * 1000.0

        return {
            "original_text": original_text,
            "repaired_text": text,
            "latency_ms": latency_ms,
        }


if __name__ == "__main__":
    print("=" * 65)
    print("TESTING CONTEXTUAL INTENT & DYSFLUENCY REPAIR")
    print("=" * 65)

    # 1. Test Without Patient Memory (Pure Stutter Cleaning)
    standalone_repair = IntentRepair()
    raw_stutter = "he really crucified him he he nailed it for a yard less"
    res1 = standalone_repair.repair(raw_stutter)

    print("\n[>] Test 1: Stutter Removal & Grammar Polishing (No Patient Memory)")
    print(f"    Original : '{res1['original_text']}'")
    print(f"    Repaired : '{res1['repaired_text']}'")
    print(f"    Latency  : {res1['latency_ms']:.2f} ms")

    # 2. Test WITH Patient Memory (FeedbackEngine attached)
    fb_engine = FeedbackEngine(patient_id="test_patient_001")
    personalized_repair = IntentRepair(feedback_engine=fb_engine)
    res2 = personalized_repair.repair(raw_stutter)

    print("\n[>] Test 2: Full Repair (Stutter Removed + Patient Memory Applied)")
    print(f"    Original : '{res2['original_text']}'")
    print(f"    Repaired : '{res2['repaired_text']}'")
    print(f"    Latency  : {res2['latency_ms']:.2f} ms")
    print("=" * 65)



if __name__ == "__main__":
    print("=" * 65)
    print("TESTING CONTEXTUAL INTENT & DYSFLUENCY REPAIR")
    print("=" * 65)

    # 1. Test Without Patient Memory (Pure Stutter Cleaning)
    standalone_repair = IntentRepair()
    raw_stutter = "he really crucified him he he nailed it for a yard less"
    res1 = standalone_repair.repair(raw_stutter)

    print("\n[>] Test 1: Stutter Removal & Grammar Polishing (No Patient Memory)")
    print(f"    Original : '{res1['original_text']}'")
    print(f"    Repaired : '{res1['repaired_text']}'")
    print(f"    Latency  : {res1['latency_ms']:.2f} ms")

    # 2. Test WITH Patient Memory (FeedbackEngine attached)
    fb_engine = FeedbackEngine(patient_id="test_patient_001")
    personalized_repair = IntentRepair(feedback_engine=fb_engine)
    res2 = personalized_repair.repair(raw_stutter)

    print("\n[>] Test 2: Full Repair (Stutter Removed + Patient Memory Applied)")
    print(f"    Original : '{res2['original_text']}'")
    print(f"    Repaired : '{res2['repaired_text']}'")
    print(f"    Latency  : {res2['latency_ms']:.2f} ms")
    print("=" * 65)
    