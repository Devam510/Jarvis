# Part 4 — Memory Architecture, Autonomy Engine, Vision Upgrade, and Observability

---

## §10 Memory Architecture

### 10.1 Dual-Layer Memory Model

```
┌──────────────────────────────────────────────┐
│  SESSION MEMORY (in-process, dict-based)     │
│  ─────────────────────────────────────────── │
│  • Current conversation turns                │
│  • Active plan state                         │
│  • Tool execution results (this session)     │
│  • TTL: process lifetime                     │
│  • Max entries: 200                          │
└──────────────────────┬───────────────────────┘
                       │ persist_on_session_end
                       ▼
┌──────────────────────────────────────────────┐
│  LONG-TERM VECTOR MEMORY (ChromaDB)          │
│  ─────────────────────────────────────────── │
│  • Conversation summaries                    │
│  • User preferences and facts                │
│  • Task outcomes                             │
│  • Skill learnings                           │
│  • Indexed by embedding similarity           │
│  • TTL: indefinite (with pruning policy)     │
└──────────────────────────────────────────────┘
```

### 10.2 Memory Schema

```python
@dataclass
class MemoryEntry:
    id: str                     # uuid4
    content: str                # natural language content
    embedding: list[float]      # 384-dim from all-MiniLM-L6-v2
    entry_type: MemoryType      # FACT, PREFERENCE, CONVERSATION, TASK_OUTCOME, SKILL
    importance: float           # 0.0–1.0, influences retrieval priority
    access_count: int           # incremented on retrieval
    last_accessed: float        # epoch timestamp
    created_at: float           # epoch timestamp
    source_correlation_id: str  # links to originating interaction
    metadata: dict              # arbitrary key-value (e.g., {"person": "Mark"})
```

### 10.3 Embedding Generation

```python
from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()
```

### 10.4 ChromaDB Integration

```python
import chromadb

class LongTermMemory:
    def __init__(self, persist_dir: str, embedding_engine: EmbeddingEngine):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="jarvis_memory",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedding_engine

    def store(self, entry: MemoryEntry):
        self.collection.upsert(
            ids=[entry.id],
            embeddings=[entry.embedding],
            documents=[entry.content],
            metadatas=[{
                "type": entry.entry_type.value,
                "importance": entry.importance,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
                "created_at": entry.created_at,
                "source": entry.source_correlation_id,
                **entry.metadata
            }]
        )

    def query(self, text: str, top_k: int = 5,
              type_filter: list[str] = None) -> list[MemoryEntry]:
        embedding = self.embedder.embed(text)
        where_filter = None
        if type_filter:
            where_filter = {"type": {"$in": type_filter}}

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        return self._parse_results(results)
```

### 10.5 Retrieval Threshold Tuning

```python
RETRIEVAL_CONFIG = {
    "max_distance": 0.75,          # cosine distance threshold — reject beyond this
    "importance_weight": 0.3,      # blend: (1-w)*similarity + w*importance
    "recency_boost": 0.1,          # boost score for entries accessed in last 24h
    "max_context_entries": 5,       # maximum memories injected into LLM prompt
    "max_context_tokens": 1500,     # token budget for memory context
}

def rank_memories(results, config=RETRIEVAL_CONFIG):
    ranked = []
    for doc, meta, distance in results:
        if distance > config["max_distance"]:
            continue
        similarity = 1.0 - distance
        importance = meta.get("importance", 0.5)
        recency = config["recency_boost"] if _is_recent(meta["last_accessed"]) else 0.0
        score = (
            (1 - config["importance_weight"]) * similarity
            + config["importance_weight"] * importance
            + recency
        )
        ranked.append((score, doc, meta))
    ranked.sort(reverse=True)
    return ranked[:config["max_context_entries"]]
```

### 10.6 Memory Pruning Policy

```python
class MemoryPruner:
    """
    Runs as a background coroutine. Periodically prunes low-value memories.
    """
    MAX_ENTRIES = 10000
    PRUNE_INTERVAL_HOURS = 24
    MIN_AGE_DAYS = 30
    MIN_ACCESS_COUNT = 2
    MIN_IMPORTANCE = 0.2

    async def prune_loop(self, memory: LongTermMemory):
        while True:
            await asyncio.sleep(self.PRUNE_INTERVAL_HOURS * 3600)
            all_entries = memory.collection.get(include=["metadatas"])
            candidates = []
            for id_, meta in zip(all_entries["ids"], all_entries["metadatas"]):
                age_days = (time.time() - meta["created_at"]) / 86400
                if (age_days > self.MIN_AGE_DAYS
                    and meta["access_count"] < self.MIN_ACCESS_COUNT
                    and meta["importance"] < self.MIN_IMPORTANCE):
                    candidates.append(id_)

            if len(all_entries["ids"]) - len(candidates) < self.MAX_ENTRIES * 0.5:
                candidates = candidates[:len(all_entries["ids"]) - int(self.MAX_ENTRIES * 0.5)]

            if candidates:
                memory.collection.delete(ids=candidates)
```

---

## §11 Autonomy Engine

### 11.1 Goal Decomposition Loop

```
USER GOAL
    │
    ▼
┌─────────────┐
│  DECOMPOSE   │ ── LLM breaks goal into sub-tasks
└─────┬───────┘
      ▼
┌─────────────┐
│    PLAN      │ ── order sub-tasks, identify dependencies
└─────┬───────┘
      ▼
┌─────────────────┐     ┌───────────┐
│ EXECUTE STEP N  │────►│ EVALUATE  │
└─────────────────┘     └─────┬─────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
     [SUCCESS]          [PARTIAL]           [FAILURE]
     next step          re-plan             retry or abort
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ▼
                        ┌──────────┐
                        │ COMPLETE │ ── summarize results
                        └──────────┘
```

### 11.2 Implementation

```python
class AutonomyEngine:
    MAX_ITERATIONS = 15
    MAX_RETRIES_PER_STEP = 3

    async def execute_goal(self, goal: str, context: list[str]) -> GoalResult:
        plan = await self.decompose(goal, context)
        results = []
        iteration = 0

        while iteration < self.MAX_ITERATIONS and plan.pending_steps:
            iteration += 1
            step = plan.next_step()

            # Execute
            try:
                result = await self.execute_step(step)
                results.append(result)
            except StepFailure as e:
                if step.retry_count < self.MAX_RETRIES_PER_STEP:
                    step.retry_count += 1
                    plan.re_enqueue(step)
                    continue
                else:
                    results.append(StepResult(status="failed", error=str(e)))
                    # Re-plan: ask LLM if remaining steps are still viable
                    plan = await self.re_plan(goal, results, plan.remaining_steps)
                    continue

            # Self-reflect
            reflection = await self.self_reflect(goal, results, plan.remaining_steps)
            if reflection.should_re_plan:
                plan = await self.re_plan(goal, results, plan.remaining_steps)

        return GoalResult(
            status="completed" if not plan.pending_steps else "partial",
            steps_completed=len([r for r in results if r.status == "success"]),
            steps_failed=len([r for r in results if r.status == "failed"]),
            total_iterations=iteration,
            results=results
        )

    async def self_reflect(self, goal, results, remaining) -> Reflection:
        prompt = f"""Goal: {goal}
Completed steps: {json.dumps([r.to_dict() for r in results])}
Remaining steps: {json.dumps([s.to_dict() for s in remaining])}

Evaluate:
1. Are completed steps achieving the goal?
2. Do remaining steps need adjustment based on results so far?
3. Should any remaining steps be skipped or modified?

Respond JSON: {{"should_re_plan": bool, "reasoning": str}}"""

        response = await self.ollama.chat(
            model=self.model,
            messages=[{"role": "system", "content": "You are a task evaluator."},
                      {"role": "user", "content": prompt}],
            format="json"
        )
        return Reflection(**json.loads(response))
```

### 11.3 Partial Completion Reporting

When `MAX_ITERATIONS` is reached or a critical step fails irrecoverably:

```python
async def report_partial_completion(self, goal_result: GoalResult):
    summary = f"Completed {goal_result.steps_completed} of "
    summary += f"{goal_result.steps_completed + goal_result.steps_failed} steps. "
    if goal_result.steps_failed > 0:
        summary += f"{goal_result.steps_failed} steps failed. "
    summary += "Partial results are available."

    await self.tts.speak(summary)
    await self.memory.store(MemoryEntry(
        content=f"Task '{goal_result.goal}': {summary}",
        entry_type=MemoryType.TASK_OUTCOME,
        importance=0.8
    ))
```

---

## §12 Vision Upgrade

### 12.1 LLaVA Integration via Ollama

```python
class VisionEngine:
    MODEL = "llava:13b-v1.6-q4_K_M"

    async def analyze_screenshot(self, screenshot_path: str,
                                  question: str = None) -> VisionResult:
        with open(screenshot_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = question or "Describe what you see on screen. Identify UI elements, text, buttons, and current application state."

        response = await self.ollama.chat(
            model=self.MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }],
            stream=False
        )
        return VisionResult(
            description=response["message"]["content"],
            screenshot_path=screenshot_path,
            timestamp=time.time()
        )
```

### 12.2 Screenshot Capture

```python
import mss

class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()

    def capture_primary(self, output_path: str = None) -> str:
        monitor = self.sct.monitors[1]  # primary monitor
        screenshot = self.sct.grab(monitor)

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".png")

        mss.tools.to_png(screenshot.rgb, screenshot.size, output=output_path)
        return output_path

    def capture_region(self, x: int, y: int, w: int, h: int) -> str:
        region = {"left": x, "top": y, "width": w, "height": h}
        screenshot = self.sct.grab(region)
        path = tempfile.mktemp(suffix=".png")
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=path)
        return path
```

### 12.3 Screen-Based Planning Integration

```python
async def vision_augmented_planning(self, transcript: str):
    """For UI-related tasks, capture screen state before planning."""
    if self._requires_vision(transcript):
        screenshot = self.screen_capture.capture_primary()
        vision_result = await self.vision_engine.analyze_screenshot(
            screenshot, question=f"The user said: '{transcript}'. Describe the current screen state relevant to this request."
        )
        # Inject vision context into LLM prompt
        augmented_transcript = (
            f"{transcript}\n\n[SCREEN CONTEXT]: {vision_result.description}"
        )
        return await self.planner.plan(augmented_transcript, self.memory_context)
    return await self.planner.plan(transcript, self.memory_context)
```

---

## §13 Observability and Governance

### 13.1 Structured Logging

```python
import structlog

logger = structlog.get_logger()
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.WriteLoggerFactory(
        file=open("logs/jarvis.jsonl", "a")
    )
)
```

All log entries include:
- `correlation_id` — traces a single interaction end-to-end
- `component` — source module
- `event` — structured event type
- `timestamp` — ISO-8601
- `level` — DEBUG/INFO/WARNING/ERROR/CRITICAL

### 13.2 Execution Trace

```python
@dataclass
class ExecutionTrace:
    correlation_id: str
    stages: list[TraceStage]

@dataclass
class TraceStage:
    stage_name: str          # e.g., "wake_detection", "stt", "planning", "risk", "execution"
    started_at: float
    completed_at: float
    input_summary: str       # truncated input for audit
    output_summary: str      # truncated output for audit
    status: str              # "success", "failure", "skipped"
    metadata: dict

class TraceCollector:
    def __init__(self):
        self.traces: dict[str, ExecutionTrace] = {}

    def start_stage(self, correlation_id: str, stage_name: str, input_summary: str):
        if correlation_id not in self.traces:
            self.traces[correlation_id] = ExecutionTrace(correlation_id, [])
        self.traces[correlation_id].stages.append(TraceStage(
            stage_name=stage_name,
            started_at=time.time(),
            completed_at=0,
            input_summary=input_summary[:500],
            output_summary="",
            status="in_progress",
            metadata={}
        ))

    def complete_stage(self, correlation_id: str, output_summary: str, status: str):
        trace = self.traces[correlation_id]
        trace.stages[-1].completed_at = time.time()
        trace.stages[-1].output_summary = output_summary[:500]
        trace.stages[-1].status = status
```

### 13.3 Risk Audit Log

```python
class RiskAuditLog:
    def __init__(self, log_path: str = "logs/risk_audit.jsonl"):
        self.log_path = log_path

    def record(self, classification: RiskClassification, authorization: AuthResult,
               execution_result: Optional[dict] = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": classification.correlation_id,
            "tool_name": classification.tool_call.tool_name,
            "arguments_hash": hashlib.sha256(
                json.dumps(classification.tool_call.arguments).encode()
            ).hexdigest(),
            "heuristic_score": classification.heuristic_score,
            "llm_risk_score": classification.llm_risk_score,
            "assigned_tier": classification.tier.value,
            "authorization": authorization.status.value,
            "execution_status": execution_result.get("status") if execution_result else None,
            "reasoning": classification.reasoning
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

### 13.4 Crash Recovery and State Restoration

```python
class StateCheckpointer:
    CHECKPOINT_INTERVAL_SECONDS = 30

    async def checkpoint_loop(self, state_machine, memory, plan_state):
        while True:
            await asyncio.sleep(self.CHECKPOINT_INTERVAL_SECONDS)
            checkpoint = {
                "timestamp": time.time(),
                "state": state_machine.current_state.value,
                "session_memory": memory.session.to_dict(),
                "active_plan": plan_state.to_dict() if plan_state else None,
                "pending_tool_calls": [tc.to_dict() for tc in state_machine.pending_calls]
            }
            with open("state/checkpoint.json", "w") as f:
                json.dump(checkpoint, f)

    def restore(self) -> Optional[dict]:
        try:
            with open("state/checkpoint.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
```
