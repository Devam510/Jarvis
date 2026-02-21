# Part 1 — Architectural Philosophy, System Architecture, and Layer Separation

---

## §1 Architectural Philosophy

### 1.1 Agent-Oriented Design vs Chatbot Design

A chatbot is a stateless request-response system. An agent is a persistent, goal-driven entity with:

- **Persistent state** across interaction boundaries
- **Autonomous goal pursuit** — the agent may decompose a user goal into sub-goals and execute them without further prompting
- **Tool-mediated world interaction** — every side-effect is performed through a typed, validated tool interface
- **Self-monitoring** — the agent evaluates its own outputs, detects failures, and re-plans

The Jarvis system is an **agent**, not a chatbot. It maintains working memory, long-term memory, environmental context, and an execution plan that persists across turns. It does not merely respond — it acts, observes, and adapts.

### 1.2 Tool-Centric Reasoning Model

The LLM never directly interacts with the operating system, filesystem, network, or any external resource. Every action is mediated through a **Tool Schema Registry** — a set of strongly-typed function declarations that the LLM may invoke via structured output.

```
LLM Output → JSON Tool Call → Schema Validator → Risk Classifier → Permission Gate → Executor
```

The LLM is an **intent generator**. It produces structured proposals. It never executes.

### 1.3 Deterministic Execution Layer Separated from Probabilistic Reasoning Layer

```
┌──────────────────────────────────┐
│   PROBABILISTIC LAYER            │
│   (LLM reasoning, planning,     │
│    intent extraction, NLU)       │
│   ── outputs structured JSON ──  │
├──────────────────────────────────┤
│   DETERMINISTIC LAYER            │
│   (schema validation, risk       │
│    classification, permission    │
│    gate, executor, audit log)    │
│   ── fully predictable ──       │
└──────────────────────────────────┘
```

The probabilistic layer (LLM) **proposes**. The deterministic layer **validates, authorizes, executes, and logs**. This separation is non-negotiable. No LLM output is trusted without deterministic validation.

### 1.4 Fail-Closed Safety Principle

When any component encounters ambiguity, malformed input, timeout, or unexpected state:

- **Default action: DENY / ABORT**
- No action is taken on ambiguous authorization
- No destructive operation proceeds without explicit positive confirmation
- All failures are logged with full context before aborting

The system never fails into an unsafe state. The default state is always "do nothing."

### 1.5 Least-Privilege Execution Model

Every tool executor runs with the minimum permissions required for its operation:

| Executor | Privilege Scope |
|---|---|
| File Reader | Read-only access to user-approved directories |
| File Writer | Write access to explicitly allowlisted paths only |
| Browser Automation | Sandboxed browser profile, no access to default user profile |
| Code Sandbox | Docker container with no network, restricted FS mount, CPU/memory caps |
| System Control | Pre-approved command allowlist only; no arbitrary shell execution |

Privilege escalation requires explicit user confirmation regardless of LLM confidence.

### 1.6 Observability-First Engineering

Every component emits structured telemetry:

- **Structured logs** (JSON, not plaintext) with correlation IDs
- **Execution traces** — full causal chain from user utterance to final action
- **Performance metrics** — latency histograms for each pipeline stage
- **Risk audit log** — every risk classification decision with inputs and outputs
- **State snapshots** — periodic serialization of agent state for crash recovery

No component operates as a black box. All internal decisions are reconstructable from logs.

---

## §2 Full System Architecture

### 2.1 Event-Driven Asynchronous Core

The system runs on a single-process, multi-coroutine `asyncio` event loop. Rationale:

- Audio capture, LLM inference, and TTS playback are I/O-bound and benefit from cooperative scheduling
- CPU-bound work (STT inference, embedding generation) is offloaded to `ProcessPoolExecutor` or dedicated threads with GIL-releasing C extensions
- Event-driven design enables natural backpressure handling — if the LLM is slow, the audio buffer accumulates without blocking

```python
class AgentEventLoop:
    """Top-level asyncio event loop orchestrator."""

    def __init__(self):
        self.event_bus = AsyncEventBus()
        self.state_machine = InteractionStateMachine()
        self.components: dict[str, Component] = {}

    async def run(self):
        await asyncio.gather(
            self.audio_subsystem.run(),
            self.state_machine.run(),
            self.watchdog.run(),
            self.memory_compactor.run(),
        )
```

### 2.2 Multi-Component Orchestration Model

Components communicate exclusively through the `AsyncEventBus` — a typed publish-subscribe system. No component holds a direct reference to another. This enables:

- **Hot-swappable** components (e.g., swap STT engine without restarting)
- **Fault isolation** — a crashed component does not cascade
- **Testing** — components can be tested in isolation by injecting mock events

```
EventBus channels:
  audio.raw_frames       → WakeWordEngine, VADEngine
  audio.wake_detected    → StateMachine
  audio.speech_segment   → STTEngine
  stt.transcript         → IntentReasoner
  cognition.tool_calls   → RiskClassifier
  risk.classified        → PermissionGate
  permission.authorized  → ExecutionEngine
  execution.result       → ResponseGenerator, MemoryManager
  tts.audio_chunk        → AudioOutput
  system.health          → Watchdog
  system.error           → ErrorHandler, AuditLog
```

### 2.3 State Machine Interaction Model

```
                    ┌──────────┐
                    │   IDLE   │◄──────────────────────────────────┐
                    └────┬─────┘                                   │
                         │ wake_word_detected                      │
                    ┌────▼─────┐                                   │
                    │ LISTENING│                                    │
                    └────┬─────┘                                   │
                         │ end_of_speech (VAD)                     │
                    ┌────▼──────┐                                  │
                    │TRANSCRIBING│                                  │
                    └────┬──────┘                                   │
                         │ transcript_ready                        │
                    ┌────▼─────┐                                   │
                    │ REASONING│                                    │
                    └────┬─────┘                                   │
                         │ tool_calls_generated                    │
                    ┌────▼─────┐                                   │
                    │ PLANNING │──── self_critique_fail ───►REASONING
                    └────┬─────┘                                   │
                         │ plan_approved                           │
                    ┌────▼──────────┐                              │
                    │RISK_CLASSIFYING│                              │
                    └────┬──────────┘                               │
                         │                                         │
              ┌──────────┼──────────┐                              │
              ▼          ▼          ▼                               │
         [Tier 1]   [Tier 2]   [Tier 3]                            │
         auto-exec  notify     ┌────▼───────────┐                  │
              │        │       │AWAITING_CONFIRM │                  │
              │        │       └────┬────────────┘                  │
              │        │            │ confirmed/denied/timeout      │
              │        │       ┌────▼──────┐                       │
              ▼        ▼       │ EXECUTING │◄──────────────────    │
              └────────┬───────┘───────────┘                       │
                       │ execution_complete                        │
                  ┌────▼──────┐                                    │
                  │ RESPONDING│                                    │
                  └────┬──────┘                                    │
                       │ response_delivered                        │
                  ┌────▼───────┐                                   │
                  │MEMORY_UPDATE│───────────────────────────────────┘
                  └────────────┘
```

States are modeled as an `enum` with guarded transitions. Invalid transitions raise `IllegalStateTransitionError` and trigger the watchdog.

### 2.4 Inter-Module Communication

All inter-module communication uses typed dataclasses:

```python
@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    confidence: float
    language: str
    timestamp: float
    correlation_id: str

@dataclass(frozen=True)
class ToolCallEvent:
    tool_name: str
    arguments: dict[str, Any]
    correlation_id: str
    source_transcript: str

@dataclass(frozen=True)
class RiskClassification:
    tool_call: ToolCallEvent
    tier: RiskTier  # Enum: TIER_1, TIER_2, TIER_3
    heuristic_score: float
    llm_risk_score: float
    reasoning: str
    correlation_id: str
```

### 2.5 Fault Boundaries

Each major component runs within a fault boundary that catches exceptions and emits structured error events:

```python
class FaultBoundary:
    def __init__(self, component_name: str, event_bus: AsyncEventBus, max_retries: int = 3):
        self.component_name = component_name
        self.event_bus = event_bus
        self.max_retries = max_retries
        self.consecutive_failures = 0

    async def execute(self, coro):
        try:
            result = await asyncio.wait_for(coro, timeout=self.timeout)
            self.consecutive_failures = 0
            return result
        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            await self.event_bus.emit("system.error", ComponentTimeout(
                component=self.component_name,
                consecutive_failures=self.consecutive_failures
            ))
            if self.consecutive_failures >= self.max_retries:
                await self.event_bus.emit("system.health", ComponentDegraded(
                    component=self.component_name,
                    action="restart_requested"
                ))
        except Exception as e:
            await self.event_bus.emit("system.error", ComponentError(
                component=self.component_name,
                error=str(e),
                traceback=traceback.format_exc()
            ))
```

### 2.6 Watchdog Supervisor Process

The watchdog runs as a **separate process** (not coroutine) that monitors the main agent process:

```python
class WatchdogSupervisor:
    """
    Runs as an independent process. Monitors the agent via:
    1. Heartbeat file — agent writes timestamp every 5s
    2. Health socket — agent responds to /health pings
    3. Resource monitoring — CPU/memory/disk thresholds
    """

    def monitor_loop(self):
        while True:
            if not self.check_heartbeat():
                self.escalate(action="restart_agent")
            if self.check_resource_exhaustion():
                self.escalate(action="force_gc_or_restart")
            if self.check_stuck_state():
                self.escalate(action="force_state_reset")
            time.sleep(self.poll_interval)

    def escalate(self, action: str):
        match action:
            case "restart_agent":
                self.kill_agent()
                self.restore_state_from_snapshot()
                self.start_agent()
            case "force_gc_or_restart":
                self.signal_agent(signal.SIGUSR1)  # trigger GC
            case "force_state_reset":
                self.signal_agent(signal.SIGUSR2)  # reset to IDLE
```

### 2.7 Explicit Pipeline

```
Audio Stream (16kHz mono, int16)
  → Wake Word Engine (Porcupine/OpenWakeWord)
    → VAD (Silero) — confirms speech, suppresses false wake
      → STT (Faster-Whisper) — streaming transcription
        → Intent Reasoner (Ollama / Llama 3)
          → Tool Planner — decomposes into tool call sequence
            → Risk Classifier — static heuristic + LLM assessment
              → Permission Gate — tier-based authorization
                → Execution Engine — sandboxed tool execution
                  → Memory Update — session + long-term persistence
                    → Response Generator — natural language response
                      → Streaming TTS (Piper) — audio output
                        → Idle (return to wake word listening)
```

---

## §3 Deterministic vs Non-Deterministic Layer Separation

### 3.1 Strict Boundary Definition

```
┌─────────────────────────────────────────────────────┐
│  NON-DETERMINISTIC (Probabilistic)                  │
│                                                     │
│  • LLM intent extraction                            │
│  • LLM tool call proposal                           │
│  • LLM risk reasoning (secondary)                   │
│  • LLM response generation                          │
│  • LLM multi-step planning                          │
│                                                     │
│  OUTPUT: JSON conforming to ToolCallSchema           │
├─────────────────────────────────────────────────────┤
│  DETERMINISTIC (Validated)                          │
│                                                     │
│  • JSON schema validation (jsonschema)              │
│  • Argument type checking                           │
│  • Path canonicalization and allowlist check         │
│  • Static risk heuristic scoring                    │
│  • Permission tier evaluation                       │
│  • Executor dispatch                                │
│  • Audit logging                                    │
│  • Memory write                                     │
│  • State machine transition                         │
└─────────────────────────────────────────────────────┘
```

### 3.2 Schema Validation

Every tool call from the LLM must conform to a registered JSON Schema:

```python
TOOL_SCHEMA_REGISTRY = {
    "file_read": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1, "maxLength": 4096},
            "encoding": {"type": "string", "enum": ["utf-8", "latin-1", "ascii"], "default": "utf-8"}
        },
        "required": ["path"],
        "additionalProperties": False
    },
    "file_write": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1, "maxLength": 4096},
            "content": {"type": "string", "maxLength": 1048576},
            "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"}
        },
        "required": ["path", "content"],
        "additionalProperties": False
    },
    "execute_code": {
        "type": "object",
        "properties": {
            "language": {"type": "string", "enum": ["python", "bash"]},
            "code": {"type": "string", "maxLength": 65536},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300, "default": 30}
        },
        "required": ["language", "code"],
        "additionalProperties": False
    }
}
```

### 3.3 Type Enforcement and Sanitization

```python
class ToolCallValidator:
    def validate(self, tool_name: str, arguments: dict) -> ValidatedToolCall:
        if tool_name not in TOOL_SCHEMA_REGISTRY:
            raise UnknownToolError(tool_name)

        schema = TOOL_SCHEMA_REGISTRY[tool_name]
        jsonschema.validate(instance=arguments, schema=schema)

        # Path sanitization for file operations
        if "path" in arguments:
            resolved = Path(arguments["path"]).resolve()
            if not self._is_within_allowed_roots(resolved):
                raise PathViolationError(resolved, self.allowed_roots)
            arguments["path"] = str(resolved)

        # Code sanitization — strip obvious injection patterns
        if "code" in arguments:
            self._check_code_safety(arguments["code"])

        return ValidatedToolCall(
            tool_name=tool_name,
            arguments=arguments,
            validated_at=time.time()
        )

    def _is_within_allowed_roots(self, path: Path) -> bool:
        return any(path.is_relative_to(root) for root in self.allowed_roots)

    def _check_code_safety(self, code: str):
        BLOCKED_PATTERNS = [
            r"import\s+subprocess", r"os\.system\(", r"eval\(",
            r"exec\(", r"__import__\(", r"open\(.*/etc/",
            r"shutil\.rmtree\(", r"os\.remove\(",
        ]
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, code):
                raise CodeSafetyViolation(pattern, code[:200])
```

### 3.4 Execution Authorization Flow

```
LLM output (JSON)
  │
  ▼
Schema Validator ──── REJECT if malformed ────► AuditLog + ErrorResponse
  │
  ▼ (ValidatedToolCall)
Risk Classifier ──── assigns RiskTier ────►
  │
  ▼ (RiskClassification)
Permission Gate
  │
  ├── Tier 1 → auto-authorize
  ├── Tier 2 → log + visual notification + auto-authorize
  └── Tier 3 → enter AWAITING_CONFIRM state
  │
  ▼ (AuthorizedToolCall)
Executor ──── performs action ────► ResultEvent
  │
  ▼
AuditLog.record(correlation_id, tool_call, risk_tier, result, timestamp)
```

Every step in this flow is deterministic, auditable, and produces a traceable artifact. The LLM is never in the execution path.
