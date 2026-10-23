"""
Dysarthria Speech Correction & Speech Gym Engine
================================================
Modules:
- synthetic_dysarthria: DSP-based neuromotor speech perturbation engine
- data_ingestion: TORGO clinical dataset ingestion and dynamic collation
- biomarker_engine: Objective bio-acoustic telemetry (Speech Gym)
- speech_recognizer: SOTA Interspeech dysarthria ASR engine (Parakeet-TDT)
- feedback_engine: Human-in-the-Loop patient verification and adaptation buffer
- intent_repair: Contextual dysfluency, stutter, and intent restoration engine
- voice_restoration: Biological voice restoration & personal timbre cloning engine
- supabase_manager: Supabase Auth, PostgreSQL telemetry, and cloud audio storage gateway
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
from .feedback_engine import FeedbackEngine
from .intent_repair import IntentRepair
from .voice_restoration import VoiceRestorationEngine
from .supabase_manager import SupabaseManager

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
    "FeedbackEngine",
    "IntentRepair",
    "VoiceRestorationEngine",
    "SupabaseManager",
]
