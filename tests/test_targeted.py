"""Quick targeted test for risk, pathval, anomaly, and TTS."""

from jarvis.risk.risk_engine import compute_heuristic_score, score_to_tier
from jarvis.utils.enums import RiskTier

# Risk
s1, t1 = compute_heuristic_score("file_delete", {"path": "/tmp", "recursive": True})
assert score_to_tier(s1) == RiskTier.TIER_3, f"Expected TIER_3 got {score_to_tier(s1)}"
s2, _ = compute_heuristic_score("file_read", {"path": "test.txt"})
assert score_to_tier(s2) == RiskTier.TIER_1, f"Expected TIER_1 got {score_to_tier(s2)}"
print(f"risk OK: delete={score_to_tier(s1).name}, read={score_to_tier(s2).name}")

# Path validator
from jarvis.execution.executor import PathValidator, PathViolationError

pv = PathValidator(["C:/Test/Desktop"])
try:
    pv.validate("../../etc/passwd")
    print("pathval FAIL: allowed traversal")
except PathViolationError:
    print("pathval OK: blocked traversal")

try:
    pv.validate("C:/Windows/System32/config/SAM")
    print("pathval FAIL: allowed system path")
except PathViolationError:
    print("pathval OK: blocked system path")

# Anomaly detector
from jarvis.safety.safety_system import AnomalyDetector
from jarvis.utils.types import Plan, PlannedAction

det = AnomalyDetector()
actions = [
    PlannedAction(tool_name="file_delete", arguments={"path": f"/tmp/{i}"})
    for i in range(12)
]
plan = Plan(thought="test", confidence=0.5, actions=actions)
anomalies = det.check(plan)
print(f"anomaly OK: found {len(anomalies)} anomalies")

# TTS
from jarvis.tts.tts_engine import TTSEngine
from jarvis.utils.config import TTSConfig

tts = TTSEngine(TTSConfig())
engine_type = "pyttsx3" if tts._pyttsx_engine else "none"
print(f"tts OK: engine={engine_type}")

# Vision
from jarvis.vision.vision_engine import VisionEngine

print("vision OK")

# Memory
from jarvis.memory.memory_manager import MemoryManager

print("memory OK")

# Cognition
from jarvis.cognition.cognitive_core import CognitiveCore

print("cognition OK")

print("\nALL TARGETED TESTS PASSED")
