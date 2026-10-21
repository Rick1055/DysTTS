"""
Speech Gym: Bio-Acoustic Biomarker Engine

Extracts objective clinical telemetry from speech audio:
1. Articulation Velocity & Pause Ratio (Motor Speed & Bradykinesia)
2. Phonatory Instability: Jitter & Shimmer (Laryngeal Ataxia)
3. Formant Centralization Ratio: FCR (Vowel Space Compression via LPC)
4. Self-Attention Entropy Probing (WavLM Transformer Probing)
"""

import numpy as np
import scipy.signal
import librosa
import torch
import math


def compute_articulation_velocity(audio, sample_rate=16000):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = audio.squeeze()

    hop_length = 512
    total_duration = len(audio)/sample_rate
    rms = librosa.feature.rms(y=audio , hop_length=512, frame_length=2048)[0]
    thres = 0.05*np.max(rms)
    active_frames = np.sum(rms > thres)
    phonation_durn = (active_frames* hop_length)/(sample_rate + 1e-7)
    phonation_durn = min(phonation_durn, total_duration)
    pause_ratio = (total_duration - phonation_durn)/ (total_duration + 1e-7)
    pause_ratio = float(np.clip(pause_ratio, 0.0, 1.0))

    envelope = np.abs(scipy.signal.hilbert(audio))
    b,a = scipy.signal.butter(N=2 , Wn=5.0, btype="low", fs=sample_rate)
    smoothed_envelope = scipy.signal.filtfilt(b, a, envelope)

    distance = int(0.15*sample_rate)
    prominence = 0.05*np.max(smoothed_envelope)
    peaks, _ = scipy.signal.find_peaks(smoothed_envelope, distance=distance, prominence=prominence)
    syllable_count = len(peaks)
    articulation_velocity = syllable_count/(phonation_durn+1e-6)
    speaking_rate = syllable_count/(total_duration + 1e-7)

    return {
        "articulation_velocity": float(articulation_velocity),
        "speaking_rate": float(speaking_rate),
        "phonation_duration": float(phonation_durn),
        "total_duration": float(total_duration),
        "pause_ratio": float(pause_ratio),
        "syllable_count": int(syllable_count)
    }

def compute_phonatory_instability(audio, sample_rate=16000):
    if isinstance(audio , torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = audio.squeeze()

    f0 , voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin = librosa.note_to_hz("C2"),
        fmax = librosa.note_to_hz("C7"),
        sr=sample_rate
    )

    valid_f0 = f0[voiced_flag & ~np.isnan(f0) & (f0 > 0)]
    periods = 1.0/valid_f0

    period_diffs = np.abs(np.diff(periods))
    jitter_pct = (np.mean(period_diffs) / (np.mean(periods) + 1e-7))*100

    rms = librosa.feature.rms(y=audio , hop_length=512)[0]
    min_len = min(len(rms), len(voiced_flag))
    voiced_amps = rms[:min_len][voiced_flag[:min_len]]
    valid_amps = voiced_amps[voiced_amps >1e-4]

    if len(valid_f0) < 3 or len(valid_amps) < 3:
        return {
            "jitter_percent":0.0,
            "shimmer_percent":0.0,
            "mean_f0": 0.0
        }

    amp_diffs = np.abs(np.diff(valid_amps))
    shimmer_pct = (np.mean(amp_diffs)/(np.mean(valid_amps) + 1e-7)) * 100

    return {
        "jitter_percent": float(jitter_pct),
        "shimmer_percent": float(shimmer_pct),
        "mean_f0": float(np.mean(valid_f0))
    }

def compute_formant_centralization(audio, sample_rate=16000):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = audio.squeeze()

    pre_emphasized = np.append(audio[0] , audio[1:] - 0.97*audio[:-1])
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    window = np.hamming(frame_length)

    f1_list = []
    f2_list = []

    for start in range(0, len(pre_emphasized) - frame_length, hop_length):
        frame = pre_emphasized[start: start + frame_length]*window

        if np.sum(frame**2) < 1e-4:
            continue

        lpc_order = 2 + int(sample_rate/1000)
        a_lpc = librosa.lpc(frame, order=lpc_order)

        roots = np.roots(a_lpc)
        roots = roots[np.imag(roots) > 0]

        angles = np.angle(roots)
        freqs = angles * (sample_rate/(2*np.pi))

        bandwidths = -0.5 * (sample_rate / (2*np.pi)) * np.log(np.abs(roots) + 1e-7)

        formants = []
        for f, bw in zip(freqs,bandwidths):
            if 200 < f <3500 and bw < 400:
                formants.append(f)

        formants = sorted(formants)

        if len(formants) >=2:
            f1,f2 = formants[0] , formants[1]
            if 200 < f1 < 1000 and 700 < f2 < 3000:
                f1_list.append(f1)
                f2_list.append(f2)

    if len(f1_list) < 5:
        return {
            "mean_f1": 500.0,
            "mean_f2": 1500.0,
            "f1_dispersion": 0.0,
            "f2_dispersion":0.0,
            "vowel_space_area": 0.0,
            "centralization_ratio": 1.0
        }

    f1_arr = np.array(f1_list)
    f2_arr = np.array(f2_list)

    mean_f1 = float(np.mean(f1_arr))
    mean_f2 = float(np.mean(f2_arr))
    std_f1 = float(np.std(f1_arr))
    std_f2 = float(np.std(f2_arr))

    vowel_space_area =float(np.pi*std_f1*std_f2)
    centralization_ratio = float(10000.0/ (vowel_space_area+1e-5))

    return {
        "mean_f1": mean_f1,
        "mean_f2": mean_f2,
        "f1_dispersion": std_f1,
        "f2_dispersion": std_f2,
        "vowel_space_area": vowel_space_area,
        "centralization_ratio": centralization_ratio
    }

_WAVLM_MODEL = None

def get_wavlm_model():
    """Lazy Loader so that the model gets loaded into the RAM only once."""
    global _WAVLM_MODEL
    if _WAVLM_MODEL is None:
        from transformers import WavLMModel
        _WAVLM_MODEL = WavLMModel.from_pretrained("microsoft/wavlm-base")
        _WAVLM_MODEL.eval()

    return _WAVLM_MODEL

def compute_attention_entropy(audio, model=None, sample_rate=16000):
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.tensor(audio , dtype=torch.float32)
    elif isinstance(audio, torch.Tensor):
        audio_tensor = audio.clone().detach().to(dtype=torch.float32)
    else:
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

    audio_tensor = audio_tensor.squeeze()
    if audio_tensor.ndim == 1:
        audio_tensor =audio_tensor.unsqueeze(0)

    if audio_tensor.ndim == 2 and audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

    if audio_tensor.shape[1] < int(0.25*sample_rate):
        return {
            "attention_entropy": 0.0,
            "phonetic_uncertainty": 0.0
        }

    wavlm = model if model is not None else get_wavlm_model()
    device = next(wavlm.parameters()).device
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        outputs = wavlm(audio_tensor, output_attentions=True)

    layer_entropies = []
    for layer_idx in range(4,8):
        att = outputs.attentions[layer_idx][0]
        T = att.shape[-1]

        entropy = -torch.sum(att * torch.log2(att+1e-9), dim = -1)
        max_entropy = math.log2(T) if T > 1 else 1.0
        norm_entropy = entropy/(max_entropy + 1e-7)
        layer_entropies.append(norm_entropy.mean().item())

    mean_entropy = float(np.mean(layer_entropies))

    return {
        "attention_entropy": mean_entropy,
        "phonetic_uncertainty": float(np.clip(mean_entropy, 0.0, 1.0))
    }

def extract_speech_gym_biomarkers(audio, sample_rate=16000, model=None):
   
    articulation = compute_articulation_velocity(audio, sample_rate=sample_rate)
    phonatory = compute_phonatory_instability(audio, sample_rate=sample_rate)
    formants = compute_formant_centralization(audio, sample_rate=sample_rate)
    attention = compute_attention_entropy(audio, model=model, sample_rate=sample_rate)

    # Calculate an overall Speech Gym Composite Score (0 to 100 scale)
    # Higher = healthier voice; Lower = severe motor impairment
    score = 100.0
    score -= min(30.0, phonatory["jitter_percent"] * 3.0)   # High jitter penalty
    score -= min(25.0, articulation["pause_ratio"] * 35.0)  # Pause struggle penalty
    score -= min(25.0, formants["centralization_ratio"] * 0.05) # Vowel collapse penalty
    composite_score = float(np.clip(score, 10.0, 100.0))

    return {
        "composite_score": composite_score,
        "articulation": articulation,
        "phonatory_instability": phonatory,
        "vowel_space": formants,
        "neural_attention": attention
    }

def main():
    import soundfile as sf
    try:
        from src.data_ingestion import TorgoDataset
    except ModuleNotFoundError:
        from data_ingestion import TorgoDataset

    print("=== Testing Speech Gym Bio-Acoustic Engine ===")
    dataset = TorgoDataset("data/TORGO")

    # Pick 1 Healthy Control and 1 Dysarthric Speaker
    ctrl_sample = next(s for s in dataset if s["group"] == "female_control")
    dys_sample = next(s for s in dataset if s["speaker_id"] == "F01")

    print(f"\n[+] Analyzing Healthy Control ({ctrl_sample['speaker_id']})...")
    ctrl_report = extract_speech_gym_biomarkers(ctrl_sample["raw_audio"])

    print(f"[+] Analyzing Dysarthric Speaker ({dys_sample['speaker_id']} - {dys_sample['diagnosis']})...")
    dys_report = extract_speech_gym_biomarkers(dys_sample["raw_audio"])

    print("\n" + "=" * 65)
    print(f"{'CLINICAL BIOMARKER':<30} | {'HEALTHY (FC02)':<15} | {'DYSARTHRIC (F01)':<15}")
    print("=" * 65)
    print(f"{'Speech Gym Composite Score':<30} | {ctrl_report['composite_score']:<15.1f} | {dys_report['composite_score']:<15.1f}")
    print(f"{'Articulation Velocity (syl/s)':<30} | {ctrl_report['articulation']['articulation_velocity']:<15.2f} | {dys_report['articulation']['articulation_velocity']:<15.2f}")
    print(f"{'Pause Ratio':<30} | {ctrl_report['articulation']['pause_ratio']*100:<14.1f}% | {dys_report['articulation']['pause_ratio']*100:<14.1f}%")
    print(f"{'Jitter (Pitch Instability)':<30} | {ctrl_report['phonatory_instability']['jitter_percent']:<14.2f}% | {dys_report['phonatory_instability']['jitter_percent']:<14.2f}%")
    print(f"{'Vowel Space Area (F1/F2)':<30} | {ctrl_report['vowel_space']['vowel_space_area']:<15.0f} | {dys_report['vowel_space']['vowel_space_area']:<15.0f}")
    print(f"{'WavLM Attention Entropy':<30} | {ctrl_report['neural_attention']['attention_entropy']:<15.4f} | {dys_report['neural_attention']['attention_entropy']:<15.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()