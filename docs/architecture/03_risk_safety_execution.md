# Part 3 — Risk Classification, Critical Command Safeguard, and Execution Engine

---

## §8 Risk Classification and Critical Command Safeguard

### 8.1 Hybrid Risk Detection Model

Risk classification operates as a two-stage pipeline: a fast deterministic heuristic layer followed by an optional LLM-based contextual assessment for ambiguous cases.

```
ToolCallEvent
  │
  ▼
┌──────────────────┐     score ≥ TIER_3_THRESHOLD
│ Static Heuristic │ ──────────────────────────────► TIER_3 (immediate)
│ Layer            │
└────────┬─────────┘
         │ score < TIER_3_THRESHOLD
         │ AND score > TIER_1_THRESHOLD
         ▼
┌──────────────────┐
│ LLM Risk         │ ──► Final tier assignment
│ Assessment       │
└──────────────────┘
```

### 8.2 Static Heuristic Layer (Layer A)

Pattern-based scoring with cumulative risk accumulation:

```python
class StaticRiskHeuristic:
    RISK_PATTERNS = {
        # Pattern → (base_score, description)
        r"\bdelete\b|\bremove\b|\bunlink\b": (0.7, "deletion_operation"),
        r"\bwipe\b|\bformat\b|\bpurge\b": (0.9, "destructive_wipe"),
        r"\bsend\b|\bemail\b|\bpost\b|\bupload\b": (0.6, "data_exfiltration_risk"),
        r"\bregistry\b|\bregedit\b": (0.8, "system_registry_modification"),
        r"\bsudo\b|\badmin\b|\bsystem32\b": (0.9, "privilege_escalation"),
        r"\brecursive\b|\b-r\b|\b-rf\b": (0.5, "recursive_operation"),
        r"\bbulk\b|\ball\b|\b\*\b": (0.4, "bulk_scope"),
        r"\bexecute\b|\brun\b|\beval\b": (0.3, "code_execution"),
        r"\bmodify\b|\boverwrite\b|\breplace\b": (0.4, "modification"),
        r"\bshutdown\b|\brestart\b|\bkill\b": (0.8, "system_control"),
    }

    TOOL_BASE_RISK = {
        "file_read": 0.0, "search_files": 0.0, "system_info": 0.0,
        "file_write": 0.3, "browser_click": 0.3, "execute_code": 0.4,
        "file_delete": 0.6, "browser_navigate": 0.1,
    }

    CRITICAL_PATH_PATTERNS = [
        r"C:\\Windows", r"C:\\Program Files", r"/etc/", r"/usr/",
        r"System32", r"\.ssh", r"\.env", r"\.git/config",
    ]

    def evaluate(self, tool_call: ValidatedToolCall) -> HeuristicResult:
        score = self.TOOL_BASE_RISK.get(tool_call.tool_name, 0.5)
        triggers = []

        serialized = json.dumps(tool_call.arguments).lower()

        for pattern, (weight, desc) in self.RISK_PATTERNS.items():
            if re.search(pattern, serialized, re.IGNORECASE):
                score = max(score, weight)
                triggers.append(desc)

        if "path" in tool_call.arguments:
            for crit_path in self.CRITICAL_PATH_PATTERNS:
                if re.search(crit_path, tool_call.arguments["path"], re.IGNORECASE):
                    score = max(score, 0.85)
                    triggers.append(f"critical_path:{crit_path}")

        if tool_call.arguments.get("recursive", False):
            score = min(score + 0.2, 1.0)
            triggers.append("recursive_flag")

        tier = self._score_to_tier(score)
        return HeuristicResult(score=score, tier=tier, triggers=triggers)

    def _score_to_tier(self, score: float) -> RiskTier:
        if score < 0.2:
            return RiskTier.TIER_1
        elif score < 0.6:
            return RiskTier.TIER_2
        else:
            return RiskTier.TIER_3
```

### 8.3 LLM-Based Risk Assessment (Layer B)

Invoked only when heuristic score falls in the ambiguous range (0.2–0.6):

```python
LLM_RISK_PROMPT = """Evaluate the risk of the following tool call.

Tool: {tool_name}
Arguments: {arguments_json}
User's original request: {transcript}
Context: {memory_context}

Rate risk on a scale of 0.0 to 1.0 where:
- 0.0–0.2: Read-only, no side effects
- 0.2–0.5: Write operation, reversible
- 0.5–0.8: Potentially destructive, significant side effects
- 0.8–1.0: Irreversible, system-critical, data loss risk

Respond JSON: {{"risk_score": float, "reasoning": str, "recommended_tier": int}}
"""

async def llm_risk_assess(self, tool_call, transcript, context) -> LLMRiskResult:
    response = await self.ollama.chat(
        model=self.model,
        messages=[
            {"role": "system", "content": "You are a security risk assessor."},
            {"role": "user", "content": LLM_RISK_PROMPT.format(
                tool_name=tool_call.tool_name,
                arguments_json=json.dumps(tool_call.arguments),
                transcript=transcript,
                memory_context=context
            )}
        ],
        format="json"
    )
    return LLMRiskResult(**json.loads(response))
```

### 8.4 Tier Model

| Tier | Risk Level | Actions | Examples |
|---|---|---|---|
| **TIER_1** | Read-Only | Auto-execute, silent log | `file_read`, `search_files`, `system_info` |
| **TIER_2** | Write Non-Destructive | Visual notification + logged | `file_write` (new file), `browser_click`, `execute_code` (sandboxed) |
| **TIER_3** | Critical / Irreversible | Explicit verbal confirmation required | `file_delete` (recursive), system commands, bulk operations, data transmission |

### 8.5 Critical Command Confirmation Protocol — State Machine

```python
class ConfirmationStateMachine:
    """
    States: SUMMARIZING → ASKING → WAITING → CONFIRMED/DENIED/TIMEOUT
    """

    TIMEOUT_SECONDS = 15.0
    CONFIRM_PHRASES = {"yes", "confirm", "do it", "proceed", "go ahead", "approved"}
    DENY_PHRASES = {"no", "stop", "cancel", "abort", "don't", "negative"}

    async def run(self, action_summary: str, tts: TTSEngine, stt: STTEngine) -> ConfirmResult:
        # STATE: SUMMARIZING
        summary = f"I'm about to {action_summary}. This is a critical operation."
        await tts.speak(summary)

        # STATE: ASKING
        await tts.speak("Do you confirm? Say 'yes' to proceed or 'no' to cancel.")

        # STATE: WAITING
        try:
            response = await asyncio.wait_for(
                stt.listen_for_response(),
                timeout=self.TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            # STATE: TIMEOUT
            await tts.speak("No confirmation received. Aborting operation.")
            return ConfirmResult(
                status=ConfirmStatus.TIMEOUT,
                action_summary=action_summary,
                timestamp=time.time()
            )

        normalized = response.text.strip().lower()

        # STATE: CONFIRMED or DENIED
        if any(phrase in normalized for phrase in self.CONFIRM_PHRASES):
            if any(phrase in normalized for phrase in self.DENY_PHRASES):
                # Ambiguous — contains both confirm and deny phrases
                await tts.speak("Your response was ambiguous. Aborting for safety.")
                return ConfirmResult(status=ConfirmStatus.AMBIGUOUS)

            return ConfirmResult(status=ConfirmStatus.CONFIRMED)
        elif any(phrase in normalized for phrase in self.DENY_PHRASES):
            await tts.speak("Operation cancelled.")
            return ConfirmResult(status=ConfirmStatus.DENIED)
        else:
            # Unrecognized response — fail closed
            await tts.speak("I didn't understand your response. Aborting for safety.")
            return ConfirmResult(status=ConfirmStatus.AMBIGUOUS)
```

```
State Diagram:

    ┌────────────┐
    │ SUMMARIZING│ ── speak summary ──►
    └─────┬──────┘
          ▼
    ┌────────────┐
    │   ASKING   │ ── speak "confirm?" ──►
    └─────┬──────┘
          ▼
    ┌────────────┐
    │  WAITING   │ ── listen (15s timeout) ──►
    └──┬──┬──┬───┘
       │  │  │
       │  │  └── timeout ──► ABORTED (logged)
       │  │
       │  └── ambiguous ──► ABORTED (logged)
       │
       ├── "yes" ──► CONFIRMED ──► execute
       │
       └── "no"  ──► DENIED ──► idle (logged)
```

---

## §9 Execution Engine

### 9.1 File System Executor

```python
class FileSystemExecutor:
    def __init__(self, config: FileSystemConfig):
        self.allowed_roots = config.allowed_roots  # list[Path]
        self.max_recursive_depth = config.max_recursive_depth  # default: 5
        self.max_file_count = config.max_file_count  # default: 100
        self.max_file_size = config.max_file_size  # default: 50MB

    async def file_read(self, path: str, encoding: str = "utf-8") -> str:
        resolved = self._resolve_and_validate(path)
        if resolved.stat().st_size > self.max_file_size:
            raise FileSizeExceeded(resolved, self.max_file_size)
        return resolved.read_text(encoding=encoding)

    async def file_write(self, path: str, content: str, mode: str = "overwrite") -> dict:
        resolved = self._resolve_and_validate(path)
        if mode == "overwrite":
            resolved.write_text(content, encoding="utf-8")
        elif mode == "append":
            with open(resolved, "a", encoding="utf-8") as f:
                f.write(content)
        return {"status": "success", "path": str(resolved), "bytes_written": len(content)}

    async def file_delete(self, path: str, recursive: bool = False) -> dict:
        resolved = self._resolve_and_validate(path)
        if recursive:
            count = sum(1 for _ in resolved.rglob("*"))
            if count > self.max_file_count:
                raise FileCountExceeded(count, self.max_file_count)
        send2trash.send2trash(str(resolved))
        return {"status": "trashed", "path": str(resolved)}

    def _resolve_and_validate(self, path: str) -> Path:
        resolved = Path(path).resolve()
        # Root directory guard
        if resolved in [Path("C:/"), Path("D:/"), Path("/")]:
            raise RootDirectoryViolation(resolved)
        # Allowlist check
        if not any(resolved.is_relative_to(root) for root in self.allowed_roots):
            raise PathNotAllowed(resolved, self.allowed_roots)
        return resolved
```

### 9.2 Browser Automation Executor

```python
class BrowserExecutor:
    def __init__(self, config: BrowserConfig):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.user_data_dir = config.user_data_dir  # persistent session dir
        self.default_timeout = config.timeout_ms  # 30000ms

    async def initialize(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.user_data_dir),
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--disable-extensions",
            ],
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        self.page = self.browser.pages[0] if self.browser.pages else await self.browser.new_page()

        # Anti-detection: remove navigator.webdriver flag
        await self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

    async def navigate(self, url: str, extract_text: bool = True,
                       screenshot: bool = False, wait_for_selector: str = None) -> dict:
        response = await self.page.goto(url, timeout=self.default_timeout,
                                         wait_until="domcontentloaded")
        result = {
            "status": response.status,
            "url": self.page.url,
            "title": await self.page.title()
        }
        if wait_for_selector:
            await self.page.wait_for_selector(wait_for_selector, timeout=10000)
        if extract_text:
            result["text"] = await self.page.inner_text("body")
            result["text"] = result["text"][:10000]  # cap extraction
        if screenshot:
            path = Path(tempfile.mktemp(suffix=".png"))
            await self.page.screenshot(path=str(path), full_page=False)
            result["screenshot_path"] = str(path)
        return result

    async def click(self, selector: str, wait_after_ms: int = 1000) -> dict:
        # DOM selector resilience: try multiple strategies
        for strategy in [selector, f"text={selector}", f"[aria-label='{selector}']"]:
            try:
                await self.page.click(strategy, timeout=5000)
                await self.page.wait_for_timeout(wait_after_ms)
                return {"status": "clicked", "selector": strategy}
            except PlaywrightTimeoutError:
                continue
        raise SelectorNotFound(selector)
```

### 9.3 Code Execution Sandbox

```python
class DockerSandbox:
    IMAGE = "python:3.12-slim"
    DEFAULTS = {
        "cpu_limit": 1.0,           # 1 CPU core
        "memory_limit": "256m",      # 256MB RAM
        "network_mode": "none",      # no network access
        "timeout_seconds": 30,
        "readonly_rootfs": True,
        "pids_limit": 50,
        "tmpfs_size": "64m"
    }

    async def execute(self, language: str, code: str,
                      timeout_seconds: int = 30) -> SandboxResult:
        container_id = f"jarvis_sandbox_{uuid.uuid4().hex[:8]}"
        code_file = self._write_temp_code(code, language)

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "run",
                "--rm",
                "--name", container_id,
                "--cpus", str(self.DEFAULTS["cpu_limit"]),
                "--memory", self.DEFAULTS["memory_limit"],
                "--network", self.DEFAULTS["network_mode"],
                "--read-only",
                "--tmpfs", f"/tmp:size={self.DEFAULTS['tmpfs_size']}",
                "--pids-limit", str(self.DEFAULTS["pids_limit"]),
                "-v", f"{code_file}:/code/script.py:ro",
                self.IMAGE,
                "python", "/code/script.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                await self._kill_container(container_id)
                return SandboxResult(
                    status="timeout",
                    stdout="",
                    stderr=f"Execution exceeded {timeout_seconds}s limit",
                    exit_code=-1
                )

            return SandboxResult(
                status="completed",
                stdout=stdout.decode()[:10000],
                stderr=stderr.decode()[:5000],
                exit_code=proc.returncode
            )
        finally:
            os.unlink(code_file)

    async def _kill_container(self, container_id: str):
        proc = await asyncio.create_subprocess_exec(
            "docker", "kill", container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait()
```
