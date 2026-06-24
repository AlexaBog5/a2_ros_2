"""Mission state machine states."""

from enum import Enum, auto


class MissionState(Enum):
    """Ordered states for the autonomous survey mission."""

    CHECK_PREREQS = auto()
    STAND = auto()
    WAIT_STAND = auto()
    UNLOCK = auto()
    WALK = auto()
    RECORD_HOME = auto()
    START_EXPLORE = auto()
    EXPLORING = auto()
    INVESTIGATING = auto()
    SAVE_MAP = auto()
    NAV_HOME = auto()
    SIT_DOWN = auto()
    DONE = auto()
    FAILED = auto()
