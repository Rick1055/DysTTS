import os
import re
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile
import numpy as np


# Clinical Metadata for TORGO Speakers
def parse_speaker_info(speaker_id, metadata_lookup=None):
    """
    Dynamically infers gender and pathology from metadata.csv or speaker prefix:
    - 'FC...' -> Female Control
    - 'F...'  -> Female Dysarthric
    - 'MC...' -> Male Control
    - 'M...'  -> Male Dysarthric
    """
    spk = speaker_id.strip().upper()

    if metadata_lookup and spk in metadata_lookup:
        row = metadata_lookup[spk]
        is_dys = str(row["is_dysarthric"]).strip().lower() == "true"
        gender = row["gender"]
        group = f"{gender.lower()}_{'dysarthric' if is_dys else 'control'}"
        return {
            "gender": gender,
            "diagnosis": row["diagnosis"],
            "severity": row["severity"],
            "is_dysarthric": is_dys,
            "group": group
        }

    if spk.startswith("FC"):
        return {
            "gender": "Female",
            "is_dysarthric": False,
            "group": "female_control",
            "diagnosis": "Control",
            "severity": "Healthy"
        }
    elif spk.startswith("F"):
        return {
            "gender": "Female",
            "is_dysarthric": True,
            "group": "female_dysarthric",
            "diagnosis": "Dysarthria",
            "severity": "Unknown"
        }
    elif spk.startswith("MC"):
        return {
            "gender": "Male",
            "is_dysarthric": False,
            "group": "male_control",
            "diagnosis": "Control",
            "severity": "Healthy"
        }
    else:
        return {
            "gender": "Male",
            "is_dysarthric": True,
            "group": "male_dysarthric",
            "diagnosis": "Dysarthria",
            "severity": "Unknown"
        }


class TorgoDataset(Dataset):
    """
    PyTorch Dataset for clinical TORGO dysarthria speech corpus.
    Features:
    - Dynamic mono/16kHz conversion and peak normalization
    - Mel-spectrogram computation with customizable n_fft & hop_length
    - Clinical metadata annotation (etiology, severity, gender)
    - Prompt cleaning (removes bracketed directions like '[as in I can read]')
    - Speaker filtering for Leave-One-Speaker-Out (LOSO) cross-validation
    - Optional filtering of unintelligible ('xxx') clinical sessions
    """
    def __init__(
        self,
        root_dir,
        sample_rate=16000,
        n_mels=80,
        hop_length=512,
        n_fft=2048,
        speakers=None,
        filter_unintelligible=False
    ):
        self.root_dir = str(root_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.speakers = set(s.strip().upper() for s in speakers) if speakers else None
        self.filter_unintelligible = filter_unintelligible
        self.samples = self._index_dataset()

    def _index_dataset(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"TORGO root directory not found at: {self.root_dir}")

        meta_lookup = {}
        csv_path = os.path.join(self.root_dir, "metadata.csv")
        if os.path.exists(csv_path):
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    meta_lookup[row["speaker_id"].strip().upper()] = row

        samples = []
        for root, dirs, files in os.walk(self.root_dir):
            if os.path.basename(root) == "wav_headMic":
                session_dir = os.path.dirname(root)
                prompts_dir = os.path.join(session_dir, "prompts")

                speaker_id = os.path.basename(os.path.dirname(session_dir)).strip().upper()

                # Speaker-level filter (e.g. for LOSO evaluation)
                if self.speakers and speaker_id not in self.speakers:
                    continue

                meta = parse_speaker_info(speaker_id, metadata_lookup=meta_lookup)

                for wav_file in sorted(files):
                    if wav_file.endswith(".wav"):
                        utt_id = os.path.splitext(wav_file)[0]
                        prompt_file = os.path.join(prompts_dir, f"{utt_id}.txt")
                        wav_path = os.path.join(root, wav_file)

                        if os.path.exists(prompt_file):
                            with open(prompt_file, mode="r", encoding="utf-8") as pf:
                                raw_text = pf.read().strip()

                            # Clean bracketed notes: "read [as in I can read]" -> "read"
                            clean_text = re.sub(r"\[.*?\]", "", raw_text).strip()
                            text = clean_text if clean_text else raw_text
                            is_valid = (text.lower() != "xxx" and len(text) > 0)

                            # Skip unintelligible prompts if filter is requested
                            if self.filter_unintelligible and not is_valid:
                                continue

                            samples.append({
                                "wav_path": wav_path,
                                "prompt_path": prompt_file,
                                "text": text,
                                "raw_text": raw_text,
                                "is_valid_transcript": is_valid,
                                "gender": meta["gender"],
                                "group": meta["group"],
                                "speaker_id": speaker_id,
                                "diagnosis": meta["diagnosis"],
                                "severity": meta["severity"],
                                "is_dysarthric": meta["is_dysarthric"]
                            })

        if len(samples) == 0:
            print(f"[!] Warning: No matching audio-prompt pairs found under {self.root_dir}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Read audio file
        audio, sr = soundfile.read(sample["wav_path"])

        # Convert dual audio channels (stereo) to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to target sample rate (16 kHz for Transformers / Whisper)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # Amplitude Normalization with epsilon to avoid division by zero
        audio = audio / (np.max(np.abs(audio)) + 1e-7)

        # Compute Mel Spectrogram (80 mel filterbanks)
        mel_bin = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        # Convert power to decibels (dB), ref=np.max puts peak power at 0.0 dB
        mel_bin_db = librosa.power_to_db(mel_bin, ref=np.max)
        mel_tensor = torch.tensor(mel_bin_db, dtype=torch.float32)

        return {
            "mel": mel_tensor,
            "raw_audio": torch.tensor(audio, dtype=torch.float32),
            "text": sample["text"],
            "raw_text": sample["raw_text"],
            "is_valid_transcript": sample["is_valid_transcript"],
            "speaker_id": sample["speaker_id"],
            "gender": sample["gender"],
            "diagnosis": sample["diagnosis"],
            "severity": sample["severity"],
            "is_dysarthric": sample["is_dysarthric"],
            "group": sample["group"],
            "wav_path": sample["wav_path"]
        }


def speech_collate_fn(batch):
    """
    Custom collator for variable-length speech tensors:
    - Mel spectrograms: padded to max_mel_time with silence floor (-80.0 dB)
    - Attention masks: 1.0 for genuine speech frames, 0.0 for padded frames
    - Mel lengths: unpadded time-frame count for each utterance
    - Raw audio: padded to max_audio_len with 0.0 amplitude
    - Audio lengths: sample count for each raw audio clip
    """
    if not batch:
        return {}

    batch_size = len(batch)
    n_mels = batch[0]["mel"].shape[0]
    max_mel_time = max(item["mel"].shape[1] for item in batch)
    max_audio_len = max(item["raw_audio"].shape[0] for item in batch)

    # Initialize padded mel tensor with -80.0 dB (silence floor in dB scale)
    padded_mels = torch.full((batch_size, n_mels, max_mel_time), fill_value=-80.0, dtype=torch.float32)
    attention_masks = torch.zeros(batch_size, max_mel_time, dtype=torch.float32)
    mel_lengths = torch.zeros(batch_size, dtype=torch.long)

    # Initialize 1D raw waveforms with 0.0 (silence in linear amplitude)
    padded_audio = torch.zeros(batch_size, max_audio_len, dtype=torch.float32)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        mel = item["mel"]
        t_len = mel.shape[1]
        padded_mels[i, :, :t_len] = mel
        attention_masks[i, :t_len] = 1.0
        mel_lengths[i] = t_len

        raw_wav = item["raw_audio"]
        w_len = raw_wav.shape[0]
        padded_audio[i, :w_len] = raw_wav
        audio_lengths[i] = w_len

    texts = [item["text"] for item in batch]
    raw_texts = [item["raw_text"] for item in batch]
    speaker_ids = [item["speaker_id"] for item in batch]
    diagnoses = [item["diagnosis"] for item in batch]
    severities = [item["severity"] for item in batch]
    groups = [item["group"] for item in batch]
    is_dysarthric = torch.tensor([item["is_dysarthric"] for item in batch], dtype=torch.bool)
    is_valid_transcript = torch.tensor([item["is_valid_transcript"] for item in batch], dtype=torch.bool)
    wav_paths = [item["wav_path"] for item in batch]

    return {
        "mel": padded_mels,
        "attention_mask": attention_masks,
        "mel_lengths": mel_lengths,
        "raw_audio": padded_audio,
        "audio_lengths": audio_lengths,
        "texts": texts,
        "raw_texts": raw_texts,
        "is_valid_transcript": is_valid_transcript,
        "speaker_ids": speaker_ids,
        "diagnoses": diagnoses,
        "severities": severities,
        "groups": groups,
        "is_dysarthric": is_dysarthric,
        "wav_paths": wav_paths
    }


def main():
    data_dir = os.path.join(os.getcwd(), "data", "TORGO")
    print("=== Testing TorgoDataset & DataLoader ===")

    # Test 1: Full dataset
    dataset = TorgoDataset(root_dir=data_dir)
    print(f"Total paired clinical recordings found: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=speech_collate_fn)

    # Grab 1 batch to verify tensor shapes
    batch = next(iter(loader))
    print("\n[+] SUCCESS! Batch loaded successfully:")
    print(f"  * Mel Spectrogram Batch Shape: {batch['mel'].shape}  [Batch, 80, Max_Time]")
    print(f"  * Attention Mask Batch Shape:  {batch['attention_mask'].shape} [Batch, Max_Time]")
    print(f"  * Mel Frame Lengths:           {batch['mel_lengths'].tolist()}")
    print(f"  * Raw Audio Batch Shape:       {batch['raw_audio'].shape} [Batch, Max_Audio_Samples]")
    print(f"  * Audio Sample Lengths:        {batch['audio_lengths'].tolist()}")
    print(f"  * Speakers in batch:           {batch['speaker_ids']}")
    print(f"  * Diagnoses in batch:          {batch['diagnoses']}")
    print(f"  * Valid Transcript Flags:      {batch['is_valid_transcript'].tolist()}")
    print(f"  * Cleaned Transcripts:         {batch['texts']}")

    # Test 2: LOSO cross-validation speaker filtering test
    loso_subset = TorgoDataset(root_dir=data_dir, speakers=["F01", "FC02"])
    print(f"\n[+] LOSO Filter Test (speakers=['F01', 'FC02']): {len(loso_subset)} recordings matched")

    # Test 3: Unintelligible filter test
    clean_subset = TorgoDataset(root_dir=data_dir, filter_unintelligible=True)
    print(f"[+] Clean Prompt Filter Test (excluding 'xxx'): {len(clean_subset)} valid recordings matched (1 'xxx' excluded)")


if __name__ == "__main__":
    main()