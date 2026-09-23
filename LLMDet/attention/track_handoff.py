"""Persistent seat numbers: a student's number belongs to where they sit.

Its own module, not ``attention.tracking``: that one imports the detector
adapter and with it mmcv, and session replay has never needed mmdet. Putting
this beside the tracker would have made every CPU-only replay host install the
detector stack to draw boxes from a cache.
"""
from typing import Dict, List, Optional, Sequence, Tuple


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _centre(b: Sequence[float]) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


class SeatRegistry:
    """Maps tracker ids to seat numbers that stay put for the whole session.

    The tracker re-issues ids -- after an occlusion, a pose change, or when a
    neighbour's track takes a box -- and everything downstream (the number on
    screen, the 1 Hz history the model needs, the dwell behind an alert) was
    keyed by that id. A first fix handed a student's state to a new id only
    within 3 s of losing them, so any longer gap still produced a new number.

    Students in a classroom do not move, so this keys everything by SEAT: the
    place a student was seen. Every seat remembers its last box for the whole
    session. A new tracker id takes the free seat it best overlaps, however long
    that seat has been empty, and only a box somewhere no seat is gets a new
    number. Numbers are handed out 1, 2, 3... in order of first appearance.

    What is kept across a gap:

    * the **number**, always;
    * the **history and dwell**, only if the seat was empty for at most
      ``history_gap_s``. After a longer absence it may be a different person,
      and the model's window would splice two unrelated minutes together, so
      the seat is reported in :meth:`pop_cleared` and warms up again
      (``min_frames_for_pred`` seconds).

    Guards, each against a failure that would be silent:

    * two tracks on screen never share a seat -- a seat held by a live track is
      not offered to another;
    * a box seen for less than ``min_seat_s`` never becomes a remembered seat,
      so a detector flicker on a bag cannot claim a number and later attract a
      real student;
    * the tracker sometimes confirms a new id while the old one is still being
      reported for the same student. The new id then gets a new seat, and when
      the old one vanishes it is folded back into the original seat and the
      temporary number is withdrawn.
    """

    def __init__(self, match_iou: float = 0.3, match_centre: float = 0.5,
                 history_gap_s: float = 10.0, min_seat_s: float = 2.0,
                 fold_window_s: float = 5.0, duplicate_iou: float = 0.5,
                 anchor_alpha: float = 0.1):
        self.match_iou = float(match_iou)
        #: centre distance, as a fraction of the seat box's longer side, within
        #: which a box still counts as that seat when the overlap is too small
        #: (a student leaning or standing up moves the box more than the seat).
        self.match_centre = float(match_centre)
        self.history_gap_s = float(history_gap_s)
        self.min_seat_s = float(min_seat_s)
        self.fold_window_s = float(fold_window_s)
        #: A vacated seat whose box overlaps an OCCUPIED seat this much was a
        #: second track on the same student, not a place of its own. 0.5 because
        # (measured value removed from the handover copy)
        self.duplicate_iou = float(duplicate_iou)
        #: How fast a seat's anchor box follows its student, per update. Slow
        #: enough that a tracker id jumping to another student cannot drag the
        #: seat with it; fast enough to follow a student leaning or shifting.
        self.anchor_alpha = float(anchor_alpha)
        self._free: List[int] = []                  # withdrawn numbers, reused lowest first
        self._on: Dict[int, float] = {}             # seat -> total seconds occupied
        self._last_t: Optional[float] = None
        self._moved: List[Tuple[int, int]] = []     # (from seat, to seat) for the caller
        self._seat_of: Dict[int, int] = {}          # live tracker id -> seat
        self._box: Dict[int, List[float]] = {}      # seat -> last box
        self._seen: Dict[int, float] = {}           # seat -> last time on screen
        self._first: Dict[int, float] = {}          # seat -> first time on screen
        self._opened: Dict[int, float] = {}         # new seat -> when it was created
        self._next = 1
        self._cleared: List[int] = []
        #: (t, tracker id, seat, how): "new", "resumed", "restarted" (resumed
        #: after a gap too long to keep history) or "folded".
        self.events: List[Tuple[float, int, int, str]] = []

    # ------------------------------------------------------------------ api --

    def resolve(self, t: float, boxes: Dict[int, Sequence[float]]) -> Dict[int, int]:
        """``{tracker id: box}`` for this frame -> ``{tracker id: seat}``."""
        t = float(t)
        # The tracker also moves a LIVE id onto a different student. Traced on an
        # 11-minute upload: id 1 started on the front student (x 860-1234) and
        # ended on the back-left one (x 405-586). Keyed by id, the seat number and
        # the model's history silently followed it onto the wrong person. So a
        # live id whose box no longer fits its seat is cut loose here and
        # re-seated by position below, like any new id.
        for tid, s in list(self._seat_of.items()):
            if tid in boxes and s in self._box and \
                    self._score(boxes[tid], self._box[s]) is None:
                del self._seat_of[tid]
                self._opened.pop(s, None)
                self.events.append((t, tid, s, "jumped"))
        for tid in [i for i in self._seat_of if i not in boxes]:
            s = self._seat_of.pop(tid)
            # Only the track that opened a seat may be folded out of it; once
            # vacated, the seat is a remembered place like any other.
            self._opened.pop(s, None)
            # A seat that never lasted is a flicker, not a place a student sat.
            if self._seen.get(s, t) - self._first.get(s, t) < self.min_seat_s:
                self._forget(s)

        occupied = set(self._seat_of.values())
        # Two seats on one spot: a vacated seat on top of an occupied one. Either
        # the tracker followed one student twice (seats 1 and 2 of an 11-minute
        # upload overlapped at IoU 0.81 from 0.3 s), or an occupied seat was
        # carried onto a remembered one -- on the same upload a passer-by's seat
        # picked up a seated student at 42 s and followed them home, so that
        # student's number went from 4 to 5. The seat with more time on that
        # spot keeps the number; the live student moves to it with their history.
        for s in [s for s in self._box if s not in occupied]:
            # Not against a seat opened moments ago: that is the NEW duplicate,
            # and _fold moves it back onto this older seat, which holds the
            # longer history.
            for o in [o for o in occupied if o in self._box and o not in self._opened]:
                if _iou(self._box[s], self._box[o]) < self.duplicate_iou:
                    continue
                if self._on.get(s, 0.0) > self._on.get(o, 0.0):
                    tid = next(i for i, x in self._seat_of.items() if x == o)
                    self._seat_of[tid] = s
                    self._box[s] = self._box[o]
                    occupied.discard(o)
                    occupied.add(s)
                    self._on[s] = self._on.get(s, 0.0) + self._on.get(o, 0.0)
                    self._forget(o)
                    self._moved.append((o, s))
                    self.events.append((t, tid, s, "merged"))
                else:
                    self._forget(s)
                    self._cleared.append(s)
                break
        new = [tid for tid in boxes if tid not in self._seat_of]
        pairs = []
        for tid in new:
            for s, sb in self._box.items():
                if s not in occupied:
                    score = self._score(boxes[tid], sb)
                    if score is not None:
                        pairs.append((score, -s, tid, s))
        taken = set()
        for _, _, tid, s in sorted(pairs, reverse=True):
            if tid in self._seat_of or s in taken:
                continue
            taken.add(s)
            self._take(t, tid, s)
        for tid in new:
            if tid not in self._seat_of:
                if self._free:
                    s = self._free.pop(0)
                else:
                    s, self._next = self._next, self._next + 1
                self._seat_of[tid] = s
                self._first[s] = t
                self._opened[s] = t
                self.events.append((t, tid, s, "new"))

        self._fold(t, boxes)

        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        for tid, box in boxes.items():
            s = self._seat_of[tid]
            self._on[s] = self._on.get(s, 0.0) + dt
            box = [float(v) for v in box]
            old = self._box.get(s)
            # An anchor, not the last box: see anchor_alpha.
            self._box[s] = box if old is None else \
                [o + self.anchor_alpha * (b - o) for o, b in zip(old, box)]
            self._seen[s] = t
        for s in [s for s, o in self._opened.items() if t - o > self.fold_window_s]:
            del self._opened[s]
        return dict(self._seat_of)

    def pop_cleared(self) -> List[int]:
        """Seats whose history and dwell the caller must discard."""
        out, self._cleared = self._cleared, []
        return out

    def pop_moved(self) -> List[Tuple[int, int]]:
        """``(from seat, to seat)``: move that seat's history and dwell across."""
        out, self._moved = self._moved, []
        return out

    # ------------------------------------------------------------- helpers --

    def _score(self, box, seat_box) -> Optional[float]:
        ov = _iou(box, seat_box)
        if ov >= self.match_iou:
            return 1.0 + ov                        # any overlap match beats distance
        # The SMALLER of the two boxes sets the tolerance. Measured against the
        # seat alone, a passer-by's large box reached a seated student 300 px away.
        side = max(1.0, min(max(seat_box[2] - seat_box[0], seat_box[3] - seat_box[1]),
                            max(box[2] - box[0], box[3] - box[1])))
        (bx, by), (sx, sy) = _centre(box), _centre(seat_box)
        d = ((bx - sx) ** 2 + (by - sy) ** 2) ** 0.5 / side
        return (1.0 - d / self.match_centre) if d <= self.match_centre else None

    def _take(self, t, tid, s):
        gap = t - self._seen.get(s, t)
        self._seat_of[tid] = s
        if gap > self.history_gap_s:
            self._cleared.append(s)
            self.events.append((t, tid, s, "restarted"))
        else:
            self.events.append((t, tid, s, "resumed"))

    def _fold(self, t, boxes):
        """Move a just-opened seat's track onto the free seat it duplicates."""
        occupied = set(self._seat_of.values())
        for tid in list(boxes):
            young = self._seat_of[tid]
            if young not in self._opened:
                continue
            best, best_score = None, None
            for s, sb in self._box.items():
                if s == young or s in occupied or s in self._opened:
                    continue
                if t - self._seen.get(s, t) > self.fold_window_s:
                    continue
                score = self._score(boxes[tid], sb)
                if score is not None and (best_score is None or score > best_score):
                    best, best_score = s, score
            if best is None:
                continue
            self._seat_of[tid] = best
            occupied.discard(young)
            occupied.add(best)
            self._forget(young)
            self._cleared.append(young)
            self.events.append((t, tid, best, "folded"))

    def _forget(self, s):
        for d in (self._box, self._seen, self._first, self._opened, self._on):
            d.pop(s, None)
        # Keep numbers compact without renumbering anyone on screen: a withdrawn
        # number is the next one handed out.
        if s not in self._free:
            self._free.append(s)
            self._free.sort()
