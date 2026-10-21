import numpy as np
import scipy.signal
import librosa
import soundfile

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
    def insert_involuntary_pauses(self, audio, num_pauses, pause_dur_range):
        if num_pauses <=0:
            return audio

        if len(audio) < self.sample_rate*1.0:
            return audio
        
        frame_rms_energy = librosa.feature.rms(y=audio , hop_length=512)[0]
        avg_energy = np.mean(frame_rms_energy)
        voice_count = np.where(frame_rms_energy >= avg_energy*0.5)[0]
        if len(voice_count) < 0.05*len(frame_rms_energy):
            return audio

        chosen_frames = np.sort(np.random.choice(voice_count, size = min(num_pauses, len(voice_count)), replace=False))
        sample_splits = chosen_frames * 512
        boundaries = [0] + list(sample_splits) + [len(audio)]

        pieces =[]
        for i in range(len(boundaries)-1):
            start = boundaries[i]
            end = boundaries[i+1]
            chunk = audio[start:end]
            pieces.append(chunk)
            if i < len(boundaries)-2:
                duration = np.random.uniform(low = pause_dur_range[0], high = pause_dur_range[1])
                pause_samples = int(duration * self.sample_rate)
                pause = np.random.normal(loc = 0 , scale=1e-4, size=pause_samples)
                pieces.append(pause)
            else:
                break
        audio = np.concatenate(pieces)
        return audio

    # Phonatory Instability (Tremor, Jitter, & Shimmer)
    def inject_phonatory_instability(self, audio, jitter_amp, shimmer_amp):
        time = np.arange(len(audio))/self.sample_rate

        # Shimmer injection
        shimmer_freq = np.random.uniform(low = 4, high = 8)
        shimmer_mask = 1.0 + shimmer_amp*np.sin(2*np.pi*shimmer_freq*time)
        audio = np.multiply(audio , shimmer_mask)

        # Jitter injection
        jitter_freq = np.random.uniform(low = 4, high = 8)
        indices = np.arange(len(audio))
        wobble = jitter_amp*np.sin(2*np.pi*jitter_freq*time) * (self.sample_rate/100.0)
        new_indices = np.clip(indices+wobble, 0, len(audio)-1)
        audio = np.interp(new_indices, indices, audio)

        return audio

    # Asthenia (Breathiness Noise)
    def add_breathiness_noise(self, audio, snr_db):
        noise = np.random.normal(loc = 0.0, scale=1.0, size=len(audio))
        b,a = scipy.signal.butter(N=4, Wn=[1000,4500], btype="bandpass", fs = self.sample_rate)
        filtered_noise = scipy.signal.lfilter(b,a,noise)

        power_audio = np.mean(np.square(audio)) + 1e-8
        noise_target_power = power_audio/ (10**(snr_db/10))

        power_noise = np.mean(np.square(filtered_noise)) + 1e-8

        scale_factor = np.sqrt(noise_target_power/power_noise)
        scaled_noise = scale_factor * filtered_noise

        envelope = np.abs(audio)/(np.max(np.abs(audio))+1e-7)
        audio = audio+(scaled_noise * envelope)
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
            'mild': {'stretch': 1.2, 'formant': 0.35, 'pauses': 1, 'jitter': 0.015, 'shimmer': 0.03, 'snr': 28.0},
            'moderate': {'stretch': 1.45, 'formant': 0.6, 'pauses': 2, 'jitter': 0.03, 'shimmer': 0.06, 'snr': 22.0},
            'severe': {'stretch': 1.75, 'formant': 0.85, 'pauses': 3, 'jitter': 0.05, 'shimmer': 0.10, 'snr': 17.0}
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
    print("=====Testing Synthetic Dysarthria Voice Generation Engine=====")

    # Loading the data
    print("Loading the clean audio sample...")
    audio_path = librosa.example("libri1")
    clean_audio , sr =librosa.load(audio_path,sr=16000)
    print(f"Clean audio loaded: {len(clean_audio)/sr:.2f}s at {sr} Hz")

    # Instantiating the Engine
    engine = SyntheticDysarthriaEngine(sample_rate=sr)

    # Applying the perturbation
    print("Applying dysarthric perturbation (severity = \"moderate\").....")
    dysarthric_audio = engine.perturb(clean_audio , severity="moderate")
    print(f"Dysarthric audio has been created: {len(dysarthric_audio)/sr:.2f}s")

    # Save both audios to .wav files
    soundfile.write("test_clean.wav", clean_audio, sr)
    soundfile.write("test_dysarthric.wav", dysarthric_audio, sr)

    print("\n[+] SUCCESS!")
    print("  1. 'test_clean.wav' saved.")
    print("  2. 'test_dysarthric.wav' saved.")

if __name__ == "__main__":
    main()








    