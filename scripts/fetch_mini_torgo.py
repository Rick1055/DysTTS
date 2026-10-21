"""
Real TORGO Mini-Dataset Extractor
=================================
Streams authentic clinical speech recordings directly from the University of
Toronto's official TORGO archives without needing to download all 18 GB.

Extracts real paired (wav_headMic/*.wav + prompts/*.txt) utterances for 10 speakers:
- 6 Dysarthric Speakers: F01, F03, F04, M01, M02, M04 (ALS & Cerebral Palsy)
- 4 Healthy Control Speakers: FC01, FC02, MC01, MC02
"""

import os
import re
import tarfile
import urllib.request

BASE_URL = "http://www.cs.toronto.edu/~complingweb/data/TORGO/"

SPEAKER_CLINICAL_INFO = {
    "F01": {"gender": "Female", "diagnosis": "Cerebral Palsy", "severity": "Moderate"},
    "F03": {"gender": "Female", "diagnosis": "Cerebral Palsy", "severity": "Severe"},
    "F04": {"gender": "Female", "diagnosis": "Cerebral Palsy", "severity": "Moderate"},
    "M01": {"gender": "Male",   "diagnosis": "Cerebral Palsy", "severity": "Severe"},
    "M02": {"gender": "Male",   "diagnosis": "ALS",            "severity": "Severe"},
    "M04": {"gender": "Male",   "diagnosis": "ALS",            "severity": "Moderate"},
    "FC01": {"gender": "Female", "diagnosis": "Control",       "severity": "Healthy"},
    "FC02": {"gender": "Female", "diagnosis": "Control",       "severity": "Healthy"},
    "MC01": {"gender": "Male",   "diagnosis": "Control",       "severity": "Healthy"},
    "MC02": {"gender": "Male",   "diagnosis": "Control",       "severity": "Healthy"},
}

ARCHIVE_PLAN = [
    ("F.tar.bz2",  ["F01", "F03", "F04"]),
    ("FC.tar.bz2", ["FC01", "FC02"]),
    ("M.tar.bz2",  ["M01", "M02", "M04"]),
    ("MC.tar.bz2", ["MC01", "MC02"]),
]


def extract_paired_torgo(target_dir, max_utterances_per_speaker=8):
    os.makedirs(target_dir, exist_ok=True)
    total_stats = {}

    print("=================================================================")
    print("  Downloading Real TORGO Mini-Dataset from University of Toronto")
    print("=================================================================")
    print(f"Target Directory: {target_dir}")
    print(f"Utterances per speaker: {max_utterances_per_speaker}")

    for archive_name, speakers in ARCHIVE_PLAN:
        archive_url = BASE_URL + archive_name
        print(f"\n--> Fetching {archive_name} for speakers: {', '.join(speakers)}")
        
        req = urllib.request.Request(archive_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            print(f"    Error opening {archive_url}: {e}")
            continue

        prompts_cache = {}  # (spk, session, utt_id) -> (text, orig_member_name)
        extracted_counts = {spk: 0 for spk in speakers}

        try:
            with tarfile.open(fileobj=resp, mode="r|bz2") as tar:
                for member in tar:
                    if not member.isfile():
                        continue

                    norm_name = member.name.replace("\\", "/")
                    parts = norm_name.split("/")

                    # Expected: {Speaker}/{Session}/{Subfolder}/{filename}
                    if len(parts) >= 4:
                        spk = parts[0]
                        session = parts[1]
                        subfolder = parts[2]
                        filename = parts[3]

                        if spk in speakers:
                            match = re.match(r"^(\d+)\.(wav|txt)$", filename)
                            if match:
                                utt_id, ext = match.groups()

                                # 1. Cache text prompts into memory
                                if subfolder == "prompts" and ext == "txt":
                                    f = tar.extractfile(member)
                                    if f:
                                        text = f.read().decode("utf-8", errors="ignore").strip()
                                        if text and not text.startswith("["):
                                            prompts_cache[(spk, session, utt_id)] = (text, norm_name)

                                # 2. Match with headMic wav and extract both simultaneously
                                elif subfolder == "wav_headMic" and ext == "wav":
                                    key = (spk, session, utt_id)
                                    if key in prompts_cache and extracted_counts[spk] < max_utterances_per_speaker:
                                        text, prompt_orig_name = prompts_cache[key]

                                        # Extract the audio file to exact path
                                        tar.extract(member, path=target_dir)

                                        # Write the paired prompt file
                                        prompt_path = os.path.join(target_dir, prompt_orig_name)
                                        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
                                        with open(prompt_path, "w", encoding="utf-8") as pf:
                                            pf.write(text)

                                        extracted_counts[spk] += 1
                                        print(f"    [{spk}] ({extracted_counts[spk]}/{max_utterances_per_speaker}) {utt_id}.wav -> \"{text}\"")

                                        # Check if all speakers in this archive are complete
                                        if all(extracted_counts[s] >= max_utterances_per_speaker for s in speakers):
                                            print(f"    ✓ All requested speakers from {archive_name} completed.")
                                            break
        except Exception as e:
            print(f"    Stream error on {archive_name}: {e}")

        total_stats.update(extracted_counts)

    print("\n=================================================================")
    print("  Mini-TORGO Real Dataset Ingestion Summary")
    print("=================================================================")
    print(f"{'Speaker':<8} | {'Gender':<7} | {'Diagnosis':<16} | {'Severity':<10} | {'Utterances'}")
    print("-" * 65)
    for spk, info in SPEAKER_CLINICAL_INFO.items():
        count = total_stats.get(spk, 0)
        print(f"{spk:<8} | {info['gender']:<7} | {info['diagnosis']:<16} | {info['severity']:<10} | {count}")
    print("=================================================================\n")


if __name__ == "__main__":
    out_dir = os.path.join(os.getcwd(), "data", "TORGO")
    extract_paired_torgo(out_dir, max_utterances_per_speaker=8)

