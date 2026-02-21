"""
jarvis.utils.config — Centralized configuration with YAML loading and defaults.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    frame_size: int = 512  # samples per frame (~32ms at 16kHz)
    buffer_seconds: float = 30.0
    pre_roll_ms: int = 800  # pre-roll to avoid first-word truncation
    max_speech_ms: int = 30000
    min_speech_ms: int = 500  # minimum speech duration before end-of-speech allowed
    silence_timeout_ms: int = 700  # shorter gap = faster response
    speech_threshold: float = 0.25  # conservative VAD threshold
    wake_word: str = "jarvis"
    target_peak: float = 0.8  # dynamic normalization target (0-1)
    noise_floor: int = 50  # int16 values below this = silence
    core_affinity: Optional[int] = None


@dataclass
class STTConfig:
    model_size: str = "small.en"
    fallback_model: str = "base.en"
    device: str = "cuda"  # "cpu" or "cuda"
    compute_type: str = "int8_float16"
    beam_size: int = 5
    language: str = "en"
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    confidence_fallback_threshold: float = 0.4
    min_confidence: float = 0.35  # below this, discard transcript
    max_retries: int = 1  # retry transcription on low confidence


@dataclass
class TTSConfig:
    model_path: str = ""  # path to piper .onnx model
    sample_rate: int = 22050
    speaker_id: int = 0
    sentence_silence: float = 0.2
    # Phase 1: Streaming TTS (edge-tts)
    edge_voice: str = "en-US-GuyNeural"
    edge_rate: str = "+0%"


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    temperature: float = 0.1
    num_predict: int = 256  # Short responses for voice commands
    num_ctx: int = 1024  # Smaller context for speed
    num_gpu: int = 99
    timeout_seconds: int = 60


@dataclass
class MemoryConfig:
    persist_dir: str = "data/chromadb"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_entries: int = 10000
    relevance_threshold: float = 0.75
    max_context_entries: int = 5
    max_context_tokens: int = 1500


@dataclass
class ExecutionConfig:
    allowed_roots: list[str] = field(
        default_factory=lambda: [
            str(Path.home() / "Desktop"),
            str(Path.home() / "Documents"),
            str(Path.home() / "Downloads"),
        ]
    )
    denied_paths: list[str] = field(default_factory=list)
    denied_drives: list[str] = field(default_factory=list)  # e.g. ["E:", "F:"]
    unrestricted_mode: bool = False  # True = blacklist-only (opt-in)
    max_recursive_depth: int = 5
    max_file_count: int = 100
    max_file_size_mb: int = 50
    browser_user_data_dir: str = "data/browser_profile"
    browser_timeout_ms: int = 30000
    sandbox_timeout_seconds: int = 30
    sandbox_memory_limit: str = "256m"
    sandbox_cpu_limit: float = 1.0
    tool_timeout_seconds: int = 30  # per-tool execution timeout
    circuit_breaker_threshold: int = 3  # consecutive failures → tool disabled


@dataclass
class StateStoreConfig:
    persist_path: str = "state/checkpoint.json"
    auto_persist_seconds: int = 30
    max_versions: int = 5  # rolling backup count


@dataclass
class SafetyConfig:
    confirmation_timeout_seconds: float = 15.0
    max_plan_iterations: int = 15
    max_retries_per_step: int = 3
    safe_mode_on_degraded: bool = True
    run_red_team_on_startup: bool = True


@dataclass
class AutonomyConfig:
    """Phase 3: Autonomous agent mode configuration."""

    # [S4] Hard iteration ceiling — overrides everything
    max_iterations: int = 10
    # [S5] Strict confidence gate — below this, stop re-planning
    confidence_threshold: float = 0.4
    # Retries per individual step
    max_retries_per_step: int = 2
    # [S3] Loop detection window — identical plans within this window = deadlock
    loop_detection_window: int = 3
    # Timeouts
    step_timeout_seconds: float = 60.0
    total_goal_timeout_seconds: float = 300.0
    # [S9] Reflection bounds
    reflection_timeout_seconds: float = 15.0
    reflection_max_tokens: int = 150
    # [S16] Resource budget per goal
    max_tool_calls_per_goal: int = 30
    max_tokens_per_goal: int = 10000
    max_execution_time_seconds: float = 300.0


# ── Phase 4: Perception & Intelligence configs ───────────────────────────


@dataclass
class PerceptionConfig:
    """Phase 4: Continuous screen perception layer."""

    enabled: bool = False  # [S1] OFF by default — explicit opt-in required
    capture_fps: float = 1.0  # frames per second
    change_threshold: float = 0.15  # pixel delta to trigger OCR analysis
    screenshot_dir: str = "data/screenshots"
    max_screenshot_disk_mb: int = 50  # [S4] hard disk usage cap
    max_screenshots: int = 10  # rolling buffer
    ocr_language: str = "en"
    max_ocr_per_minute: int = 10  # [S4] OCR call rate limit
    cpu_disable_threshold: float = 85.0  # [S4] auto-disable above this %
    cpu_check_window_seconds: float = 30.0  # sustained duration before disable
    privacy_apps: list[str] = field(  # [S2] app name blacklist
        default_factory=lambda: ["KeePass", "1Password", "Signal"]
    )
    privacy_title_patterns: list[str] = field(  # [S2] window title regex
        default_factory=lambda: [
            r"(?i)bank",
            r"(?i)password",
            r"(?i)auth",
            r"(?i)login",
            r"(?i)otp",
            r"(?i)vault",
        ]
    )


@dataclass
class ProactiveConfig:
    """Phase 4: Proactive suggestion engine."""

    enabled: bool = True
    confidence_threshold: float = 0.8
    cooldown_seconds: float = 600.0  # 10 min between similar triggers
    max_suggestions_per_hour: int = 6
    do_not_disturb: bool = False  # [S5] global DND flag


@dataclass
class BehavioralMemoryConfig:
    """Phase 4: Action pattern learning."""

    enabled: bool = True
    persist_dir: str = "data/behavior_db"
    min_pattern_count: int = 3  # need 3+ occurrences to predict
    prediction_threshold: float = 0.7
    max_patterns: int = 5000
    debounce_seconds: float = 5.0  # [S3] min time between writes
    max_writes_per_minute: int = 6  # [S3] hard write cap
    batch_flush_interval: float = 10.0  # [S3] batch flush period


# ── Phase 5: Reliability & Determinism configs ───────────────────────────


@dataclass
class WatchdogConfig:
    """Phase 5: Deadlock and loop detection watchdogs."""

    enabled: bool = True
    loop_detect_window: int = 20  # plan steps to track for cycle detection
    loop_cycle_threshold: int = 3  # repeated cycles before abort
    max_action_seconds: float = 60.0  # per-action timeout
    max_state_seconds: float = 90.0  # max time in non-IDLE without transition
    check_interval: float = 5.0  # watchdog tick interval


@dataclass
class MetricsConfig:
    """Phase 5: Structured metrics collection."""

    enabled: bool = True
    flush_interval_seconds: int = 60
    output_path: str = "logs/metrics.jsonl"
    max_histogram_size: int = 1000  # max samples per histogram


# ── Phase 6: Enterprise Safety & Governance configs ──────────────────────


@dataclass
class GuardrailConfig:
    """Phase 6: Deterministic guardrail limits."""

    enabled: bool = True
    max_file_deletes_per_min: int = 50
    max_file_ops_per_min: int = 200
    max_code_exec_per_min: int = 20
    domain_whitelist: list = field(
        default_factory=lambda: [
            "google.com",
            "github.com",
            "stackoverflow.com",
            "wikipedia.org",
            "python.org",
            "pypi.org",
        ]
    )


@dataclass
class AuditConfig:
    """Phase 6: Cryptographic audit ledger."""

    enabled: bool = True
    ledger_path: str = "logs/audit.ledger"


@dataclass
class MemoryGovernanceConfig:
    """Phase 6: Memory privacy and TTL."""

    pii_enabled: bool = True
    ttl_hours: float = 24.0
    ttl_check_interval: float = 300.0


# ── Future Modules configs ────────────────────────────────────────────────


@dataclass
class FastSearchConfig:
    """Everything SDK fast file search."""

    enabled: bool = False
    es_exe_path: str = "es.exe"  # path to Everything CLI
    timeout_seconds: float = 3.0  # hard timeout
    max_results: int = 50  # default cap
    absolute_max_results: int = 200  # hard ceiling


@dataclass
class CoPilotConfig:
    """Live Code Co-Pilot guardrails."""

    enabled: bool = False
    max_retries: int = 3  # hard auto-fix ceiling
    run_timeout_seconds: float = 30.0  # per-run timeout
    token_budget: int = 4096  # max tokens per session
    tool_call_budget: int = 5  # max tool calls per run
    allow_network: bool = False  # block network by default
    require_confirm_overwrite: bool = True  # TIER_3 for file overwrite


@dataclass
class ProcessGraphConfig:
    """System State Graph / ProcessMonitor."""

    enabled: bool = False
    collection_interval: float = 10.0  # seconds between collections
    min_interval: float = 5.0  # hard minimum
    cpu_anomaly_threshold: float = 80.0  # % for anomaly
    mem_anomaly_threshold: float = 85.0  # % for anomaly
    disk_anomaly_threshold: float = 90.0  # % for anomaly
    anomaly_sustained_seconds: float = 300.0  # 5 min sustained


@dataclass
class GUIConfig:
    """PySide6 GUI Dashboard."""

    enabled: bool = False
    mode: str = "developer"  # minimal | developer | admin
    theme: str = "dark"  # dark | light
    window_width: int = 1200
    window_height: int = 800


@dataclass
class BargeInConfig:
    """Barge-in detection during TTS."""

    enabled: bool = True
    vad_threshold: float = 0.6
    debounce_ms: int = 300


@dataclass
class PluginConfig:
    """Dynamic plugin marketplace."""

    enabled: bool = False
    plugins_dir: str = "plugins"


@dataclass
class ResearchConfig:
    """Autonomous web research."""

    enabled: bool = False
    max_pages: int = 5
    page_timeout: float = 15.0
    total_timeout: float = 120.0


@dataclass
class SelfHealerConfig:
    """Self-healing supervisor."""

    enabled: bool = False
    stt_latency_threshold_ms: float = 2000.0
    llm_latency_threshold_ms: float = 5000.0
    tts_latency_threshold_ms: float = 3000.0
    error_rate_threshold: float = 0.3
    check_interval: float = 30.0


@dataclass
class ObservabilityConfig:
    log_dir: str = "logs"
    log_file: str = "jarvis.jsonl"
    risk_audit_file: str = "risk_audit.jsonl"
    trace_dir: str = "logs/traces"
    checkpoint_file: str = "state/checkpoint.json"
    checkpoint_interval_seconds: int = 30


@dataclass
class JarvisConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    state_store: StateStoreConfig = field(default_factory=StateStoreConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)
    # Phase 4: Perception & Intelligence
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    behavioral_memory: BehavioralMemoryConfig = field(
        default_factory=BehavioralMemoryConfig
    )
    # Phase 5: Reliability & Determinism
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    # Phase 6: Enterprise Safety & Governance
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    memory_governance: MemoryGovernanceConfig = field(
        default_factory=MemoryGovernanceConfig
    )
    # Future modules
    fast_search: FastSearchConfig = field(default_factory=FastSearchConfig)
    copilot: CoPilotConfig = field(default_factory=CoPilotConfig)
    process_graph: ProcessGraphConfig = field(default_factory=ProcessGraphConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    # Batch 2 future modules
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    self_healer: SelfHealerConfig = field(default_factory=SelfHealerConfig)
    debug: bool = False

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "JarvisConfig":
        """Load configuration from YAML file, falling back to defaults."""
        config = cls()
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            config = cls._merge(config, data)
        return config

    @classmethod
    def _merge(cls, config: "JarvisConfig", data: dict) -> "JarvisConfig":
        """Merge dict data into config dataclass recursively."""
        for section_name, section_data in data.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                if isinstance(section_data, dict) and hasattr(
                    section, "__dataclass_fields__"
                ):
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
                else:
                    setattr(config, section_name, section_data)
        return config

    def ensure_dirs(self):
        """Create all required data directories."""
        dirs = [
            self.memory.persist_dir,
            self.observability.log_dir,
            self.observability.trace_dir,
            self.execution.browser_user_data_dir,
            os.path.dirname(self.observability.checkpoint_file) or "state",
            os.path.dirname(self.state_store.persist_path) or "state",
            # Phase 4
            self.perception.screenshot_dir,
            self.behavioral_memory.persist_dir,
            # Phase 5
            os.path.dirname(self.metrics.output_path) or "logs",
            # Phase 6
            os.path.dirname(self.audit.ledger_path) or "logs",
            "state/transactions",
            "state/tx_backups",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
