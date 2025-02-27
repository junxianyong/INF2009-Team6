from enum import Enum


class GateStatus(Enum):
    OPENED = 1
    CLOSED = 2
    FACE = 3
    VOICE = 4
    DIFF = 5


class MantrapStatus(Enum):
    IDLE = 1
    SCAN = 2
    CHECKED = 3
    ALERT = 4
    MULTI = 5
