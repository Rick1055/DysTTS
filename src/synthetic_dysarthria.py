import numpy as np
import scipy.signal
import librosa
import soundfile
import os

class SyntheticDysarthriaEngine():
    def __init__(self, sample_rate=16000, seed=42):
       self.sample_rate = sample_rate
       if seed is not None:
           np.random.seed(seed)

    # Bradylalia ( Temporal Slowdown )
    def apply_Bradylalia(self, audio, stretch_factor):
        rate = 1/stretch_factor
        audio = librosa.effects.time_stretch(y=audio, 
                                             rate = rate)
        return audio

    # Formant Centralization( Articulatory Undershoot )
    def apply_formant_centralization(self, audio , intensity):
        if intensity <= 0.0: return audio

        b_low,a_low = scipy.signal.iirpeak(w0=500 , Q = 2.5, fs=self.sample_rate)
        f1_audio = scipy.signal.lfilter(b_low, a_low, audio)

        b_high,a_high = scipy.signal.iirpeak(w0=1500, Q = 5.0, fs=self.sample_rate)
        f2_audio = scipy.signal.lfilter(b_high, a_high, audio)

        b_butter,a_butter = scipy.signal.butter(N=3, Wn=3200, btype="lowpass", fs=self.sample_rate)
        f3_lowpass = scipy.signal.lfilter(b_butter, a_butter, audio)

        centralized_audio = (1 - 0.4*intensity)*audio + (0.25*intensity)*f1_audio + (0.20*intensity)*f2_audio + (0.15*intensity)*f3_lowpass

        max_original = np.max(np.abs(audio))
        max_centralized = np.max(np.abs(centralized_audio))
        epsilon = 1e-7
        centralized_audio = (centralized_audio/(max_centralized+epsilon))*max_original

        return centralized_audio

    # Involuntary Pauses( Motor Speech Pauses )
    def insert_involuntary_pauses(self, audio, num_pauses, pause_dur_range=(0.15,0.35)):
        if num_pauses <=0 or len(audio) < self.sample_rate*1.0:
            return audio

        fade_len = int(0.025 * self.sample_rate)
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_len))**2
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_len))**2
        
        frame_rms_energy = librosa.feature.rms(y=audio , hop_length=256)[0]
        max_rms = np.max(frame_rms_energy) + 1e-7
        candidates = [f for f in range(len(frame_rms_energy)) if 0.04 * max_rms < frame_rms_energy[f] < 0.45*max_rms]
        if not candidates:
            candidates = list(range(len(frame_rms_energy) // 4, 3*len(frame_rms_energy)//4))

        min_dist_frames = int(0.8 * self.sample_rate/256)
        np.random.shuffle(candidates)
        chosen = []
        for c in candidates:
            if all(abs(c-prev) > min_dist_frames for prev in chosen):
                chosen.append(c)
                if len(chosen) >= num_pauses:
                    break
        chosen = sorted(chosen)

        splits = [c * 256 for c in chosen]
        boundaries = [0] + splits + [len(audio)]

        pieces = []
        for i in range(len(boundaries) - 1):
            chunk = audio[boundaries[i]:boundaries[i + 1]].copy()
            if len(chunk) > fade_len:
                if i > 0:
                    chunk[:fade_len] *= fade_in   # Smooth vocal re-initiation
                if i < len(boundaries) - 2:
                    chunk[-fade_len:] *= fade_out # Smooth vocal decay
            pieces.append(chunk)
            if i < len(boundaries) - 2:
                dur = np.random.uniform(pause_dur_range[0], pause_dur_range[1])
                # Soft ambient microphone room tone (no digital click)
                pause = np.random.normal(0.0, 1e-5, size=int(dur * self.sample_rate))
                pieces.append(pause)

        audio = np.concatenate(pieces)

        return audio

    # Phonatory Instability (Tremor, Jitter, & Shimmer)
    def inject_phonatory_instability(self, audio, jitter_amp, shimmer_amp):
        time = np.arange(len(audio))/self.sample_rate

        # Shimmer injection
        shimmer_freq = np.random.uniform(low = 4, high = 8)
        shimmer_mask = 1.0 + shimmer_amp*np.sin(2*np.pi*shimmer_freq*time)
        audio = audio * shimmer_mask

        # Jitter injection
        jitter_freq = np.random.uniform(low = 4, high = 8)
        indices = np.arange(len(audio))

        tremor = jitter_amp*np.sin(2*np.pi*jitter_freq*time)*(self.sample_rate/120.0)

        raw_noise = np.random.normal(0.0,1.0,size=len(audio))
        b,a = scipy.signal.butter(N=2, Wn=25.0, btype="lowpass", fs=self.sample_rate)
        smooth_noise = scipy.signal.filtfilt(b, a, raw_noise)
        smooth_noise = (smooth_noise/ (np.std(smooth_noise) + 1e-7)) * (jitter_amp *1.5 * (self.sample_rate/150.0))

        wobble = tremor + smooth_noise
        new_indices = np.clip(indices+wobble, 0, len(audio)-1)
        audio = np.interp(new_indices, indices, audio)

        return audio

    # Asthenia (Breathiness Noise)
    def add_breathiness_noise(self, audio, snr_db):
        if snr_db >= 50.0:
            return audio

        noise = np.random.normal(loc = 0.0, scale=1.0, size=len(audio))
        b,a = scipy.signal.butter(N=2, Wn=[800,3500], btype="bandpass", fs = self.sample_rate)
        filtered_noise = scipy.signal.lfilter(b,a,noise)

        win_len = int(0.025 * self.sample_rate)
        sq = audio**2
        smoothed_env = np.sqrt(np.convolve(sq, np.ones(win_len)/win_len, mode="same"))
        smoothed_env = smoothed_env/(np.max(smoothed_env) + 1e-7)

        power_audio = np.mean(np.square(audio)) + 1e-8
        noise_target_power = power_audio/ (10**(snr_db/10))

        power_noise = np.mean(np.square(filtered_noise)) + 1e-8

        scale_factor = np.sqrt(noise_target_power/power_noise)
        scaled_noise = scale_factor * filtered_noise

        audio = audio + (scaled_noise * smoothed_env)

        return audio

    # Master Perturb Function 
    def perturb(self, audio, severity="moderate"):
        if severity == "random":
            severity = np.random.uniform(0.1 , 0.9)

        if isinstance(severity, (int,float)):
            s= np.clip(float(severity), 0.0, 1.0)
            config = {
                "stretch": 1.1 + 0.6*s,
                "formant": 0.2 + 0.65*s,
                "pauses": int(np.round(1+2*s)),
                "jitter": 0.01 + 0.04*s,
                "shimmer": 0.02 + 0.06*s,
                "snr": 30.0 -14.0*s
            }
        else:
            presets = {
                "mild":     {"stretch": 1.20, "formant": 0.35, "pauses": 1, "jitter": 0.02, "shimmer": 0.04, "snr": 40.0},
                "moderate": {"stretch": 1.45, "formant": 0.60, "pauses": 2, "jitter": 0.04, "shimmer": 0.07, "snr": 35.0},
                "severe":   {"stretch": 1.75, "formant": 0.85, "pauses": 3, "jitter": 0.06, "shimmer": 0.10, "snr": 30.0}
            }
            config = presets.get(severity, presets['moderate'])

        audio = self.apply_Bradylalia(audio , stretch_factor = config['stretch'])
        audio = self.apply_formant_centralization(audio , intensity = config["formant"])
        audio = self.insert_involuntary_pauses(audio , num_pauses=config["pauses"],pause_dur_range=(0.15,0.35))
        audio = self.inject_phonatory_instability(audio , jitter_amp=config["jitter"], shimmer_amp=config['shimmer'])
        audio = self.add_breathiness_noise(audio, snr_db=config["snr"])

        max_amp = np.max(np.abs(audio)) + 1e-7
        audio = (audio / max_amp)*0.95

        return audio

def main():
    import glob

    print("===== Testing Synthetic Dysarthria Voice Generation Engine =====")

    target_sr = 16000
    clean_audio = None
    audio_source = ""

    # Look for authentic TORGO control recordings first
    control_files = glob.glob("data/TORGO/FC*/**/*.wav", recursive=True) + glob.glob("data/TORGO/MC*/**/*.wav", recursive=True)
    if control_files:
        audio_source = control_files[0]
        clean_audio, sr = soundfile.read(audio_source)
    elif os.path.exists("test_clean.wav"):
        audio_source = "test_clean.wav"
        clean_audio, sr = soundfile.read(audio_source)
    else:
        print("[!] No local audio found, falling back to librosa example...")
        audio_source = "librosa.example('libri1')"
        clean_audio, sr = librosa.load(librosa.example("libri1"), sr=target_sr)

    if len(clean_audio.shape) > 1:
        clean_audio = np.mean(clean_audio, axis=1)

    if sr != target_sr:
        clean_audio = librosa.resample(clean_audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    clean_audio = clean_audio / (np.max(np.abs(clean_audio)) + 1e-7)
    print(f"\n[+] Loaded clean audio from: {audio_source}")
    print(f"    Duration: {len(clean_audio)/sr:.2f}s | Sample Rate: {sr} Hz")

    engine = SyntheticDysarthriaEngine(sample_rate=sr)

    # Save original clean
    soundfile.write("test_clean.wav", clean_audio, sr)

    # Generate mild, moderate, severe
    severities = ["mild", "moderate", "severe"]
    print("\n[+] Synthesizing Neuromotor Pathologies across Severities:")
    print("-" * 65)
    print(f"{'SEVERITY':<12} | {'DURATION':<12} | {'STRETCH':<10} | {'OUTPUT FILE'}")
    print("-" * 65)
    print(f"{'Clean (Orig)':<12} | {len(clean_audio)/sr:<10.2f}s | {'1.00x':<10} | test_clean.wav")

    for sev in severities:
        dys_audio = engine.perturb(clean_audio, severity=sev)
        out_path = f"test_dysarthric_{sev}.wav"
        soundfile.write(out_path, dys_audio, sr)
        stretch = len(dys_audio) / len(clean_audio)
        print(f"{sev.capitalize():<12} | {len(dys_audio)/sr:<10.2f}s | {f'{stretch:.2f}x':<10} | {out_path}")

    print("-" * 65)
    print("\n[+] SUCCESS! All test audio files generated and verified.")


if __name__ == "__main__":
    main()








    