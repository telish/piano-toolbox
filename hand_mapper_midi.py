"""Kalman filter based hand mapping."""

from collections import namedtuple
import math

import scipy.stats


MidiEvent = namedtuple("MidiEvent", ["pitch", "is_note_on", "when"])


class HandConstraints(object):

    def __init__(self):
        self.sounding_notes = [False] * 128
        self.right_hand_notes = []
        self.left_hand_notes = []

    def right_hand(self):
        if self.right_hand_notes is None:
            self._assign_notes()
        return self.right_hand_notes

    def left_hand(self):
        if self.left_hand_notes is None:
            self._assign_notes()
        return self.left_hand_notes

    def midi_event(self, event):
        self.right_hand_notes = None
        self.left_hand_notes = None
        self.sounding_notes[event.pitch] = event.is_note_on

    def _assign_notes(self):
        comfortable_hand_span = 14  # 14 semitones = a ninth
        self.right_hand_notes = []
        self.left_hand_notes = []

        lowest = self._lowest_note()
        if lowest == 127:
            return

        highest = self._highest_note()
        if highest == 0:
            return

        for i in range(128):
            if self.sounding_notes[i]:
                if (i <= lowest + comfortable_hand_span) and (
                    i < highest - comfortable_hand_span
                ):
                    self.left_hand_notes.append(i)
                elif (i > lowest + comfortable_hand_span) and (
                    i >= highest - comfortable_hand_span
                ):
                    self.right_hand_notes.append(i)

    def _lowest_note(self):
        for i in range(128):
            if self.sounding_notes[i]:
                return i
        return 127

    def _highest_note(self):
        for i in reversed(range(128)):
            if self.sounding_notes[i]:
                return i
        return 0


class KalmanMapper(object):
    def __init__(self):
        self.left_hand_pos = 43.0  # mLeftHandPosition
        self.right_hand_pos = 77.0
        self.left_hand_variance = 1000.0
        self.right_hand_variance = 1000.0
        self.hand_constraints = HandConstraints()

        self.time_last_rh = None
        self.time_last_lh = None
        self.last_was_left_hand = False  # the result
        self.saved_result = []

    def midi_event(self, event):
        variance_per_second = 20.0
        midi_variance = 20.0

        self.hand_constraints.midi_event(event)
        if not event.is_note_on:
            return

        assign_left = False
        for p in self.hand_constraints.left_hand():
            if p == event.pitch:
                assign_left = True
                self.saved_result.append(("left", 1.0))

        assign_right = False
        if not assign_left:
            for p in self.hand_constraints.right_hand():
                if p == event.pitch:
                    assign_right = True
                    self.saved_result.append(("right", 1.0))

        if not assign_left and not assign_right:
            delta_rh = abs(self.right_hand_pos - event.pitch) / math.sqrt(
                self.right_hand_variance
            )
            delta_lh = abs(self.left_hand_pos - event.pitch) / math.sqrt(
                self.left_hand_variance
            )
            assign_right = delta_lh > delta_rh
            assign_left = not assign_right

            if assign_left:
                normal = scipy.stats.norm(
                    self.left_hand_pos, math.sqrt(self.left_hand_variance)
                )
                p = normal.pdf(event.pitch)
                assert p <= 1
                self.saved_result.append(("left", p))
            else:
                normal = scipy.stats.norm(
                    self.right_hand_pos, math.sqrt(self.right_hand_variance)
                )
                p = normal.pdf(event.pitch)
                assert p <= 1
                self.saved_result.append(("right", p))
            self.last_was_left_hand = assign_left

        if assign_left:
            if self.time_last_lh is not None:
                delta = event.when - self.time_last_lh
                self.left_hand_variance += delta * variance_per_second

            self.left_hand_pos += (
                self.left_hand_variance
                / (self.left_hand_variance + midi_variance)
                * (event.pitch - self.left_hand_pos)
            )
            self.left_hand_variance -= (
                self.left_hand_variance
                / (self.left_hand_variance + midi_variance)
                * self.left_hand_variance
            )
            self.time_last_lh = event.when
            self.last_was_left_hand = True

        if assign_right:
            if self.time_last_rh:
                delta = event.when - self.time_last_rh
                self.right_hand_variance += delta * variance_per_second

            self.right_hand_pos += (
                self.right_hand_variance
                / (self.right_hand_variance + midi_variance)
                * (event.pitch - self.right_hand_pos)
            )
            self.right_hand_variance -= (
                self.right_hand_variance
                / (self.right_hand_variance + midi_variance)
                * self.right_hand_variance
            )
            self.time_last_rh = event.when
            self.last_was_left_hand = False


if __name__ == "__main__":
    mapper = KalmanMapper()
    event = MidiEvent(72, True, 0.0)
    mapper.midi_event(event)
    print(event)
    print("Left hand:", mapper.last_was_left_hand)

    event = MidiEvent(40, True, 0.5)
    mapper.midi_event(event)
    print(event)
    print("Left hand:", mapper.last_was_left_hand)
