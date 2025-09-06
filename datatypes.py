from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import numpy.typing as npt

HandLiteral = Literal["left", "right", ""]


class MidiResult(TypedDict):
    type: Literal["note_on", "note_off"]
    pitch: int
    velocity: int
    hand: HandLiteral
    fingers: list[int]


class CorrespondingPoints(TypedDict):
    """Corresponding points for homography calculation."""

    pixel: tuple[int, int]
    object: tuple[float, float] | None


@dataclass
class TrackingResult:
    left_visible: bool = False
    right_visible: bool = False
    left_landmarks_xyz: tuple[list[float], list[float], list[float]] = field(default_factory=lambda: ([], [], []))
    right_landmarks_xyz: tuple[list[float], list[float], list[float]] = field(default_factory=lambda: ([], [], []))


Image = npt.NDArray[Any]
