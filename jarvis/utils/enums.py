"""
jarvis.utils.enums — All enumerations used across the Jarvis system.
"""

from enum import Enum, auto


class InteractionState(Enum):
    """State machine states for the agent interaction lifecycle."""

    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    REASONING = auto()
    PLANNING = auto()
    RISK_CLASSIFYING = auto()
    AWAITING_CONFIRM = auto()
    EXECUTING = auto()
    RESPONDING = auto()
    MEMORY_UPDATE = auto()
    ERROR = auto()
    # Phase 3: Autonomous multi-step execution states
    AUTONOMOUS_PLANNING = auto()
    AUTONOMOUS_EXECUTING = auto()


class AudioState(Enum):
    """States for the audio capture loop."""

    WAITING_FOR_WAKE = auto()
    COLLECTING_SPEECH = auto()


class RiskTier(Enum):
    """Risk classification tiers."""

    TIER_1 = 1  # Read-only — auto-execute
    TIER_2 = 2  # Write non-destructive — notify + log
    TIER_3 = 3  # Critical / irreversible — requires confirmation


class ConfirmStatus(Enum):
    """Confirmation dialogue outcomes."""

    CONFIRMED = auto()
    DENIED = auto()
    TIMEOUT = auto()
    AMBIGUOUS = auto()


class MemoryType(Enum):
    """Categories of long-term memory entries."""

    FACT = "fact"
    PREFERENCE = "preference"
    CONVERSATION = "conversation"
    TASK_OUTCOME = "task_outcome"
    SKILL = "skill"


class ComponentStatus(Enum):
    """Health status of a component."""

    HEALTHY = auto()
    DEGRADED = auto()
    FAILED = auto()
    RESTARTING = auto()


class EventPriority(Enum):
    """Priority levels for event bus backpressure (Phase 5).

    Lower numeric value = higher priority.
    P0/P1 events are NEVER dropped under backpressure.
    P3 events are dropped first when the bus is overloaded.
    """

    CRITICAL = 0  # Voice input, safety signals — always dispatched
    HIGH = 1  # Confirmation requests — always dispatched
    NORMAL = 2  # Execution, cognition — queued under pressure
    BACKGROUND = 3  # Logging, metrics, perception — dropped under pressure
