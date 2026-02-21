# Part 2 — Technology Stack, Audio Subsystem, Speech Recognition, and Cognitive Core

---

## §4 Core Local Technology Stack

### 4.1 Component Selection and Justification

| Component | Technology | Justification |
|---|---|---|
| **LLM** | Ollama + Llama 3.1 8B (Q4_K_M) | Local inference, function-calling capable, 4-bit quantization fits 8GB VRAM. Ollama provides HTTP API with streaming. |
| **STT** | Faster-Whisper (CTranslate2) | 4× faster than OpenAI Whisper, INT8 quantization, VAD filtering built-in. Runs on CPU or CUDA. |
| **Wake Word** | OpenWakeWord (primary), Porcupine (fallback) | OpenWakeWord is open-source, custom-trainable. Porcupine offers sub-100ms latency with commercial license. |
| **VAD** | Silero VAD | ONNX model, <1ms per frame, language-agnostic, no cloud dependency. |
| **TTS** | Piper | Local VITS-based synthesis, streaming chunk output, multiple voice models, <200ms first-chunk latency on CPU. |
| **Memory DB** | ChromaDB | Embedded vector database, no server process needed, supports HNSW indexing, Python-native. |
| **Browser** | Playwright | Chromium/Firefox/WebKit support, async API, persistent contexts, stealth mode via plugins. |
| **Code Sandbox** | Docker (restricted) | Process isolation, cgroup resource limits, no-network mode, disposable containers. |
| **File Ops** | os/shutil/send2trash wrappers | send2trash for soft-delete, shutil for copies, custom wrappers for path validation. |
| **Search** | ripgrep (rg) | Fastest grep alternative, respects .gitignore, streaming output, JSON mode for structured results. |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | 384-dim embeddings, <10ms per sentence on CPU, good semantic clustering. |

### 4.2 Integration Contracts

```
┌─────────────┐    16kHz int16 frames     ┌──────────────┐
│ Microphone  │ ───────────────────────► │ AudioRouter  │
│ (sounddevice)│                          │              │
└─────────────┘                          └──┬───┬───────┘
                                            │   │
                              frames (30ms) │   │ frames (30ms)
                                            ▼   ▼
                                   ┌────────┐ ┌──────┐
                                   │WakeWord│ │ VAD  │
                                   │Engine  │ │Engine│
                                   └───┬────┘ └──┬───┘
                                       │         │
                          wake_detected│         │speech_prob
                                       ▼         ▼
                                ┌─────────────────┐
                                │ SpeechCollector  │
                                │ (circular buffer)│
                                └────────┬────────┘
                                         │ speech_segment (bytes)
                                         ▼
                                ┌─────────────────┐
                                │  Faster-Whisper  │ ──► TranscriptEvent
                                └─────────────────┘
                                         │
                                         ▼
                        ┌────────────────────────────┐
                        │  Ollama (Llama 3.1 8B)     │
                        │  HTTP POST /api/chat       │
                        │  stream=true               │
                        │  format=json               │ ──► ToolCallEvent[]
                        └────────────────────────────┘
                                         │
                                         ▼
                        ┌────────────────────────────┐
                        │  Execution Pipeline        │
                        │  (validate → classify →    │
                        │   authorize → execute)     │ ──► ResultEvent
                        └────────────────────────────┘
                                         │
                                         ▼
                        ┌────────────────────────────┐
                        │  ChromaDB                  │
                        │  .add(embedding, metadata) │ ──► MemoryUpdateEvent
                        └────────────────────────────┘
                                         │
                                         ▼
                        ┌─────────────────┐
                        │  Piper TTS      │
                        │  stdin→stdout   │
                        │  streaming WAV  │ ──► AudioOutput
                        └─────────────────┘
```

**Ollama Integration Contract:**
```python
# Request
POST http://localhost:11434/api/chat
{
    "model": "llama3.1:8b-instruct-q4_K_M",
    "messages": [...],
    "stream": true,
    "format": "json",
    "options": {
        "temperature": 0.1,
        "num_predict": 2048,
        "num_ctx": 8192
    }
}

# Response (streamed, line-delimited JSON)
{"message": {"role": "assistant", "content": "{\"tool\": \"file_read\""}, "done": false}
{"message": {"role": "assistant", "content": ", \"args\": {\"path\":"}, "done": false}
...
{"message": {"role": "assistant", "content": "}}"}, "done": true}
```

**Faster-Whisper Integration Contract:**
```python
from faster_whisper import WhisperModel

model = WhisperModel("base.en", device="cuda", compute_type="int8")
segments, info = model.transcribe(
    audio_array,           # numpy float32, 16kHz
    beam_size=5,
    language="en",
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500}
)
for segment in segments:
    yield TranscriptEvent(
        text=segment.text,
        confidence=segment.avg_logprob,
        start=segment.start,
        end=segment.end
    )
```

---

## §5 Wake Word and Audio Subsystem

### 5.1 Audio Pipeline Specification

```
Microphone → 16kHz, mono, int16 (PCM)
Frame size: 512 samples (32ms)
Buffer: Rolling circular buffer, 30 seconds capacity (480,000 samples)
Pre-roll: 500ms captured before wake word detection point
```

### 5.2 Circular Buffer Implementation

```python
class CircularAudioBuffer:
    """Lock-free circular buffer for audio frames."""

    def __init__(self, capacity_seconds: float = 30.0, sample_rate: int = 16000):
        self.capacity = int(capacity_seconds * sample_rate)
        self.buffer = np.zeros(self.capacity, dtype=np.int16)
        self.write_pos = 0
        self.total_written = 0

    def write(self, frames: np.ndarray):
        n = len(frames)
        start = self.write_pos % self.capacity
        end = start + n
        if end <= self.capacity:
            self.buffer[start:end] = frames
        else:
            first = self.capacity - start
            self.buffer[start:] = frames[:first]
            self.buffer[:n - first] = frames[first:]
        self.write_pos += n
        self.total_written += n

    def read_last(self, num_samples: int) -> np.ndarray:
        """Read the last N samples from the buffer."""
        if num_samples > self.capacity:
            num_samples = self.capacity
        end = self.write_pos % self.capacity
        start = (end - num_samples) % self.capacity
        if start < end:
            return self.buffer[start:end].copy()
        else:
            return np.concatenate([
                self.buffer[start:],
                self.buffer[:end]
            ])
```

### 5.3 Real-Time Audio Loop (Pseudocode)

```python
async def audio_capture_loop(
    event_bus: AsyncEventBus,
    buffer: CircularAudioBuffer,
    wake_engine: WakeWordEngine,
    vad_engine: VADEngine,
    sample_rate: int = 16000,
    frame_size: int = 512
):
    """
    Main audio capture loop. Runs on a dedicated thread with core affinity.
    Non-blocking async integration via janus queue.
    """
    stream = sounddevice.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='int16',
        blocksize=frame_size
    )

    speech_collector = SpeechCollector(
        pre_roll_ms=500,
        max_duration_ms=30000,
        silence_timeout_ms=1500
    )

    state = AudioState.WAITING_FOR_WAKE

    with stream:
        while True:
            frames = await read_frames_async(stream, frame_size)
            buffer.write(frames)

            if state == AudioState.WAITING_FOR_WAKE:
                if wake_engine.process(frames):
                    pre_roll = buffer.read_last(sample_rate // 2)  # 500ms
                    speech_collector.start(pre_roll)
                    state = AudioState.COLLECTING_SPEECH
                    await event_bus.emit("audio.wake_detected", WakeEvent())

            elif state == AudioState.COLLECTING_SPEECH:
                speech_prob = vad_engine.process(frames)
                speech_collector.add_frames(frames, speech_prob)

                if speech_collector.end_of_speech_detected():
                    segment = speech_collector.finalize()
                    await event_bus.emit("audio.speech_segment", segment)
                    state = AudioState.WAITING_FOR_WAKE

                elif speech_collector.max_duration_exceeded():
                    segment = speech_collector.finalize()
                    await event_bus.emit("audio.speech_segment", segment)
                    state = AudioState.WAITING_FOR_WAKE
```

### 5.4 VAD-Based End-of-Speech Detection

```python
class SpeechCollector:
    def __init__(self, pre_roll_ms, max_duration_ms, silence_timeout_ms):
        self.pre_roll_samples = int(pre_roll_ms * 16)
        self.max_samples = int(max_duration_ms * 16)
        self.silence_samples = int(silence_timeout_ms * 16)
        self.frames = []
        self.speech_probs = []
        self.silence_counter = 0
        self.total_samples = 0
        self.speech_threshold = 0.5

    def add_frames(self, frames: np.ndarray, speech_prob: float):
        self.frames.append(frames)
        self.speech_probs.append(speech_prob)
        self.total_samples += len(frames)

        if speech_prob < self.speech_threshold:
            self.silence_counter += len(frames)
        else:
            self.silence_counter = 0

    def end_of_speech_detected(self) -> bool:
        return (
            self.silence_counter >= self.silence_samples
            and self.total_samples > self.pre_roll_samples + 1600  # at least 100ms of speech
        )

    def max_duration_exceeded(self) -> bool:
        return self.total_samples >= self.max_samples

    def finalize(self) -> SpeechSegment:
        audio = np.concatenate(self.frames)
        result = SpeechSegment(
            audio=audio,
            sample_rate=16000,
            duration_ms=len(audio) / 16,
            avg_speech_prob=np.mean(self.speech_probs)
        )
        self.reset()
        return result
```

### 5.5 CPU Core Pinning

```python
import os
import psutil

def pin_audio_thread(core_id: int = 0):
    """Pin the audio capture thread to a specific CPU core for consistent latency."""
    p = psutil.Process(os.getpid())
    p.cpu_affinity([core_id])
```

---

## §6 Speech Recognition Layer

### 6.1 Model Size Tradeoffs

| Model | Parameters | VRAM (INT8) | WER (en) | RTF (CPU) | RTF (GPU) | Use Case |
|---|---|---|---|---|---|---|
| tiny.en | 39M | 75MB | 7.7% | 0.5x | 0.05x | Ultra-low latency, simple commands |
| base.en | 74M | 150MB | 5.2% | 1.0x | 0.08x | **Default — best latency/accuracy tradeoff** |
| small.en | 244M | 500MB | 3.4% | 2.5x | 0.15x | Complex queries, dictation |
| medium.en | 769M | 1.5GB | 2.9% | 6.0x | 0.30x | Noisy environments, accented speech |
| large-v3 | 1550M | 3.0GB | 2.0% | 12.0x | 0.50x | Maximum accuracy, offline batch |

**Default selection:** `base.en` with INT8 quantization. Automatic fallback to `small.en` if confidence drops below threshold.

### 6.2 Streaming Decoding

```python
async def transcribe_streaming(
    audio_segment: SpeechSegment,
    model: WhisperModel,
    event_bus: AsyncEventBus
):
    audio_float = audio_segment.audio.astype(np.float32) / 32768.0

    segments, info = model.transcribe(
        audio_float,
        beam_size=5,
        best_of=3,
        language="en",
        condition_on_previous_text=False,  # prevents hallucination loops
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,   # detect repetition hallucinations
        vad_filter=True
    )

    full_text = []
    for segment in segments:
        if segment.no_speech_prob > 0.6:
            continue  # skip non-speech segments
        if segment.avg_logprob < -1.5:
            continue  # skip low-confidence segments
        full_text.append(segment.text.strip())

    transcript = " ".join(full_text)

    if not transcript or len(transcript) < 2:
        await event_bus.emit("stt.empty_transcript", EmptyTranscript())
        return

    confidence = info.all_language_probs.get("en", 0.0) if info.all_language_probs else 0.5

    await event_bus.emit("stt.transcript", TranscriptEvent(
        text=transcript,
        confidence=confidence,
        language=info.language,
        timestamp=time.time(),
        correlation_id=str(uuid.uuid4())
    ))
```

### 6.3 Hallucination Mitigation

Whisper models hallucinate under specific conditions. Mitigations:

1. **Compression ratio filter** — if output tokens / input duration exceeds 2.4, the segment is likely repetitive hallucination. Discard.
2. **No-speech probability** — segments with `no_speech_prob > 0.6` are discarded.
3. **Log probability threshold** — segments with `avg_logprob < -1.5` are low-confidence and discarded.
4. **Disable `condition_on_previous_text`** — prevents the decoder from hallucinating based on previous segment context.
5. **Maximum segment duration cap** — segments longer than 30s are split and re-transcribed independently.

### 6.4 Re-Transcription Fallback

```python
async def transcribe_with_fallback(audio: SpeechSegment, models: dict):
    result = await transcribe_streaming(audio, models["base.en"])

    if result.confidence < 0.4 or len(result.text) < 3:
        # Retry with larger model
        result = await transcribe_streaming(audio, models["small.en"])

        if result.confidence < 0.3:
            return TranscriptEvent(
                text="[UNRECOGNIZED]",
                confidence=0.0,
                language="unknown"
            )
    return result
```

---

## §7 Cognitive Core

### 7.1 System Prompt Contract

```python
SYSTEM_PROMPT = """You are Jarvis, an autonomous desktop assistant. You execute tasks by calling tools.

RULES:
1. You MUST respond with valid JSON containing a "thought" and "actions" array.
2. Each action MUST reference a tool from the AVAILABLE TOOLS list.
3. You MUST NOT fabricate tool names or arguments not in the schema.
4. For multi-step tasks, output ALL steps in order. Each step will be executed sequentially.
5. If you are uncertain, set "confidence" below 0.5 and explain in "thought".
6. NEVER attempt to execute system commands outside the tool interface.
7. You have access to MEMORY CONTEXT below — use it to personalize responses.
8. For destructive operations (delete, overwrite, send), explicitly flag "requires_confirmation": true.

AVAILABLE TOOLS:
{tool_schemas_json}

MEMORY CONTEXT:
{retrieved_memories}

CURRENT DATETIME: {iso_timestamp}
OS: {os_info}
WORKING DIRECTORY: {cwd}
"""
```

### 7.2 Tool Schema Registry

```python
TOOL_SCHEMAS = [
    {
        "name": "file_read",
        "description": "Read contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "encoding": {"type": "string", "default": "utf-8"}
            },
            "required": ["path"]
        },
        "risk_tier": "TIER_1"
    },
    {
        "name": "file_write",
        "description": "Write or append content to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["overwrite", "append"]}
            },
            "required": ["path", "content"]
        },
        "risk_tier": "TIER_2"
    },
    {
        "name": "file_delete",
        "description": "Move a file to system trash (soft delete).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean", "default": false}
            },
            "required": ["path"]
        },
        "risk_tier": "TIER_3"
    },
    {
        "name": "execute_code",
        "description": "Execute code in a sandboxed Docker container.",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {"type": "string", "enum": ["python", "bash", "javascript"]},
                "code": {"type": "string"},
                "timeout_seconds": {"type": "integer", "default": 30}
            },
            "required": ["language", "code"]
        },
        "risk_tier": "TIER_2"
    },
    {
        "name": "browser_navigate",
        "description": "Navigate browser to a URL and optionally extract content.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "extract_text": {"type": "boolean", "default": true},
                "screenshot": {"type": "boolean", "default": false},
                "wait_for_selector": {"type": "string"}
            },
            "required": ["url"]
        },
        "risk_tier": "TIER_1"
    },
    {
        "name": "browser_click",
        "description": "Click an element in the browser by CSS selector.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {"type": "string"},
                "wait_after_ms": {"type": "integer", "default": 1000}
            },
            "required": ["selector"]
        },
        "risk_tier": "TIER_2"
    },
    {
        "name": "system_info",
        "description": "Retrieve system information (CPU, memory, disk, processes).",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["cpu", "memory", "disk", "processes", "all"]}
            },
            "required": ["category"]
        },
        "risk_tier": "TIER_1"
    },
    {
        "name": "search_files",
        "description": "Search for files matching a pattern using ripgrep.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "directory": {"type": "string"},
                "file_type": {"type": "string"},
                "max_results": {"type": "integer", "default": 50}
            },
            "required": ["pattern"]
        },
        "risk_tier": "TIER_1"
    }
]
```

### 7.3 Multi-Step Planner

```python
class MultiStepPlanner:
    async def plan(self, transcript: str, memory_context: list[str]) -> Plan:
        response = await self.ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": transcript}
            ],
            format="json",
            stream=True
        )

        parsed = json.loads(response)
        return Plan(
            thought=parsed["thought"],
            confidence=parsed.get("confidence", 0.5),
            actions=[
                PlannedAction(
                    tool_name=a["tool"],
                    arguments=a["args"],
                    requires_confirmation=a.get("requires_confirmation", False),
                    depends_on=a.get("depends_on")  # index of prior action
                )
                for a in parsed["actions"]
            ]
        )
```

### 7.4 Self-Critique Mechanism

```python
SELF_CRITIQUE_PROMPT = """Review your proposed actions for the following risks:
1. Does any action have unintended side effects?
2. Are all file paths valid and within allowed directories?
3. Could any action cause data loss?
4. Is the plan complete — does it achieve the user's goal?
5. Are there redundant or unnecessary steps?

Proposed actions:
{actions_json}

Respond with JSON: {"approved": bool, "issues": [str], "revised_actions": [...] | null}
"""

async def self_critique(self, plan: Plan) -> CritiqueResult:
    response = await self.ollama.chat(
        model=self.model,
        messages=[
            {"role": "system", "content": "You are a safety reviewer."},
            {"role": "user", "content": SELF_CRITIQUE_PROMPT.format(
                actions_json=json.dumps([a.to_dict() for a in plan.actions])
            )}
        ],
        format="json"
    )
    return CritiqueResult(**json.loads(response))
```

### 7.5 Example LLM Structured Output

User utterance: *"Find all Python files on my desktop and list them"*

```json
{
    "thought": "The user wants to find all Python files on their Desktop directory. I will use search_files with a .py pattern scoped to the Desktop path.",
    "confidence": 0.92,
    "actions": [
        {
            "tool": "search_files",
            "args": {
                "pattern": "*.py",
                "directory": "C:\\Users\\Jigar Patel\\Desktop",
                "max_results": 100
            },
            "requires_confirmation": false,
            "depends_on": null
        }
    ]
}
```

### 7.6 Memory Retrieval Injection (RAG)

```python
async def retrieve_context(self, query: str, top_k: int = 5) -> list[str]:
    results = self.chromadb_collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"type": {"$in": ["fact", "preference", "conversation"]}},
        include=["documents", "metadatas", "distances"]
    )

    relevant = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if dist < self.relevance_threshold:  # cosine distance < 0.7
            relevant.append(f"[{meta['type']}] {doc}")

    return relevant
```
