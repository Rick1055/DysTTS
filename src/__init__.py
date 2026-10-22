"""
Dysarthria Speech Correction & Speech Gym Engine
================================================
Modules:
- synthetic_dysarthria: DSP-based neuromotor speech perturbation engine
- data_ingestion: TORGO clinical dataset ingestion and dynamic collation
- biomarker_engine: Objective bio-acoustic telemetry (Speech Gym)
- speech_recognizer: SOTA Interspeech dysarthria ASR engine (Parakeet-TDT)
"""

from .synthetic_dysarthria import SyntheticDysarthriaEngine
from .data_ingestion import TorgoDataset, speech_collate_fn, parse_speaker_info
from .biomarker_engine import (
    compute_articulation_velocity,
    compute_formant_centralization,
    compute_attention_entropy,
    compute_phonatory_instability,
    extract_speech_gym_biomarkers,
)
from .speech_recognizer import DysarthriaSpeechRecognizer

__all__ = [
    "SyntheticDysarthriaEngine",
    "TorgoDataset",
    "speech_collate_fn",
    "parse_speaker_info",
    "compute_articulation_velocity",
    "compute_formant_centralization",
    "compute_attention_entropy",
    "compute_phonatory_instability",
    "extract_speech_gym_biomarkers",
    "DysarthriaSpeechRecognizer",
]

