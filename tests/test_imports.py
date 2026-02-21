"""Quick functional test of all Jarvis modules."""

import sys


def test_all():
    passed = 0
    failed = 0

    # 1. Enums
    try:
        from jarvis.utils.enums import InteractionState, RiskTier, MemoryType

        assert InteractionState.IDLE.name == "IDLE"
        print("1. enums OK")
        passed += 1
    except Exception as e:
        print(f"1. enums FAILED: {e}")
        failed += 1

    # 2. Config
    try:
        from jarvis.utils.config import JarvisConfig

        c = JarvisConfig.load("config.yaml")
        assert c.llm.model == "llama3.2:3b"
        print(f"2. config OK: model={c.llm.model}")
        passed += 1
    except Exception as e:
        print(f"2. config FAILED: {e}")
        failed += 1

    # 3. Event bus
    try:
        from jarvis.core.event_bus import AsyncEventBus

        bus = AsyncEventBus()
        print("3. event_bus OK")
        passed += 1
    except Exception as e:
        print(f"3. event_bus FAILED: {e}")
        failed += 1

    # 4. State machine
    try:
        from jarvis.core.state_machine import (
            InteractionStateMachine,
            IllegalStateTransition,
        )

        sm = InteractionStateMachine()
        sm.transition(InteractionState.LISTENING)
        assert sm.state == InteractionState.LISTENING
        try:
            sm.transition(InteractionState.EXECUTING)  # should fail
            print("4. state_machine FAILED: allowed illegal transition")
            failed += 1
        except IllegalStateTransition:
            sm.transition(InteractionState.TRANSCRIBING)
            print(f"4. state_machine OK: transitions work, illegal blocked")
            passed += 1
    except Exception as e:
        print(f"4. state_machine FAILED: {e}")
        failed += 1

    # 5. Observability
    try:
        from jarvis.observability.logger import (
            setup_logging,
            TraceCollector,
            RiskAuditLog,
        )

        print("5. observability OK")
        passed += 1
    except Exception as e:
        print(f"5. observability FAILED: {e}")
        failed += 1

    # 6. Tool schemas
    try:
        from jarvis.cognition.tool_schemas import ToolRegistry

        reg = ToolRegistry()
        assert len(reg.tools) >= 13
        print(f"6. tool_schemas OK: {len(reg.tools)} tools registered")
        passed += 1
    except Exception as e:
        print(f"6. tool_schemas FAILED: {e}")
        failed += 1

    # 7. Risk heuristics
    try:
        from jarvis.risk.risk_engine import compute_heuristic_score, score_to_tier

        score, triggers = compute_heuristic_score(
            "file_delete", {"path": "/tmp/test", "recursive": True}
        )
        assert score >= 0.7
        assert score_to_tier(score) == RiskTier.TIER_3
        score2, _ = compute_heuristic_score("file_read", {"path": "test.txt"})
        assert score_to_tier(score2) == RiskTier.TIER_1
        print(f"7. risk_heuristics OK: delete=T3, read=T1")
        passed += 1
    except Exception as e:
        print(f"7. risk_heuristics FAILED: {e}")
        failed += 1

    # 8. Path validator
    try:
        from jarvis.execution.executor import PathValidator, PathViolationError

        pv = PathValidator(["C:/Users/Test/Desktop"])
        try:
            pv.validate("../../etc/passwd")
            print("8. path_validator FAILED: allowed traversal")
            failed += 1
        except PathViolationError:
            try:
                pv.validate("C:/Windows/System32/config/SAM")
                print("8. path_validator FAILED: allowed system path")
                failed += 1
            except PathViolationError:
                print("8. path_validator OK: blocked traversal + system path")
                passed += 1
    except Exception as e:
        print(f"8. path_validator FAILED: {e}")
        failed += 1

    # 9. Anomaly detector
    try:
        from jarvis.safety.safety_system import AnomalyDetector
        from jarvis.utils.types import Plan, PlannedAction

        detector = AnomalyDetector()
        # Create suspicious plan
        actions = [
            PlannedAction(tool_name="file_delete", arguments={"path": f"/tmp/{i}"})
            for i in range(12)
        ]
        plan = Plan(thought="test", confidence=0.5, actions=actions)
        anomalies = detector.check(plan)
        assert len(anomalies) > 0
        print(
            f"9. anomaly_detector OK: found {len(anomalies)} anomalies in malicious plan"
        )
        passed += 1
    except Exception as e:
        print(f"9. anomaly_detector FAILED: {e}")
        failed += 1

    # 10. TTS engine (pyttsx3 fallback)
    try:
        from jarvis.tts.tts_engine import TTSEngine
        from jarvis.utils.config import TTSConfig

        tts = TTSEngine(TTSConfig())  # no piper model → falls to pyttsx3
        print("10. tts_engine OK (pyttsx3 fallback)")
        passed += 1
    except Exception as e:
        print(f"10. tts_engine FAILED: {e}")
        failed += 1

    # 11. Vision import
    try:
        from jarvis.vision.vision_engine import VisionEngine, ScreenCapture

        print("11. vision_engine OK")
        passed += 1
    except Exception as e:
        print(f"11. vision_engine FAILED: {e}")
        failed += 1

    # 12. Autonomy import
    try:
        from jarvis.autonomy.autonomy_engine import AutonomyEngine

        print("12. autonomy_engine OK")
        passed += 1
    except Exception as e:
        print(f"12. autonomy_engine FAILED: {e}")
        failed += 1

    # 13. Main entry point import
    try:
        from jarvis.__main__ import main, print_banner

        print("13. __main__ OK")
        passed += 1
    except Exception as e:
        print(f"13. __main__ FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(test_all())
