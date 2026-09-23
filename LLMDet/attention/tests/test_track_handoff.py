"""Seat numbers must not change when the tracker re-issues an id.

The tracker hands a student a new id after an occlusion or a pose change, and
the number on screen, the 1 Hz history the model needs and the dwell behind an
alert were all keyed by that id. `SeatRegistry` keys them by where the student
sits instead, for the whole session.

What these pin is the part that is silent when wrong in the OTHER direction: two
students on screen must never share a number, a seat that was empty for a long
time keeps its number but must not splice an old history onto a new one, and a
detector flicker must not claim a number.
"""
from attention.track_handoff import SeatRegistry

A = [100.0, 100.0, 200.0, 300.0]
A_JITTER = [102.0, 98.0, 201.0, 301.0]     # same seat, a few px of jitter
A_LEAN = [150.0, 100.0, 250.0, 300.0]      # same seat, leaning: IoU 0.33 -> 0.2 side
B = [400.0, 100.0, 500.0, 300.0]           # the next seat along
C = [700.0, 100.0, 800.0, 300.0]


def _run(reg, script):
    """script: [(t, {tracker id: box})] -> list of mappings"""
    return [reg.resolve(t, boxes) for t, boxes in script]


def test_numbers_are_handed_out_in_order_not_by_tracker_id():
    reg = SeatRegistry()
    assert reg.resolve(0.0, {17: A, 4: B, 9: C}) == {17: 1, 4: 2, 9: 3}


def test_a_new_id_in_the_same_seat_keeps_the_number_and_the_history():
    reg = SeatRegistry(history_gap_s=10.0)
    _run(reg, [(0.0, {5: A}), (15.0, {5: A}), (15.3, {})])
    assert reg.resolve(15.6, {7: A_JITTER}) == {7: 1}
    assert reg.pop_cleared() == []


def test_a_long_absence_keeps_the_number_but_restarts_the_history():
    reg = SeatRegistry(history_gap_s=10.0)
    _run(reg, [(0.0, {5: A}), (20.0, {5: A}), (20.5, {})])
    assert reg.resolve(200.0, {31: A}) == {31: 1}, "the seat number must survive any gap"
    assert reg.pop_cleared() == [1], "a minutes-old window must not be spliced on"


def test_a_leaning_student_keeps_the_seat():
    reg = SeatRegistry()
    _run(reg, [(0.0, {5: A}), (5.0, {5: A}), (5.2, {})])
    assert reg.resolve(6.0, {8: A_LEAN}) == {8: 1}


def test_two_students_on_screen_never_share_a_number():
    reg = SeatRegistry()
    for t in (0.0, 1.0, 2.0, 3.0):
        m = reg.resolve(t, {1: A, 2: B})
        assert m == {1: 1, 2: 2}
    m = reg.resolve(4.0, {1: A, 2: B, 3: A_JITTER})   # a duplicate box on seat 1
    assert len(set(m.values())) == 3


def test_a_duplicate_id_is_folded_back_when_the_old_one_vanishes():
    reg = SeatRegistry()
    _run(reg, [(0.0, {1: A, 2: B}), (4.0, {1: A, 2: B})])
    assert reg.resolve(4.5, {1: A, 2: B, 9: A_JITTER})[9] == 3
    assert reg.resolve(5.0, {2: B, 9: A_JITTER}) == {2: 2, 9: 1}
    assert 3 in reg.pop_cleared()
    assert reg.resolve(6.0, {2: B, 9: A, 11: C})[11] == 3, "the withdrawn number is reused"


def test_a_different_place_gets_a_new_number():
    reg = SeatRegistry()
    _run(reg, [(0.0, {5: A}), (5.0, {5: A}), (5.5, {})])
    assert reg.resolve(6.0, {7: B}) == {7: 2}


def test_a_flicker_never_becomes_a_seat():
    reg = SeatRegistry(min_seat_s=2.0)
    _run(reg, [(0.0, {1: A}), (0.5, {1: A, 2: B}), (1.0, {1: A})])  # B for 0.5 s
    m = reg.resolve(30.0, {1: A, 3: B})
    assert m[3] == 2 and reg.events[-1][3] == "new", \
        "a box seen for half a second must not be remembered as a place someone sat"


def test_a_second_track_on_the_same_student_does_not_leave_a_second_seat():
    """Two tracks on one student from the first frame (seen on a real upload:
    IoU 0.81). When one vanishes its seat must go, or the student's number
    depends on which of the two remembered seats wins when they return."""
    reg = SeatRegistry(duplicate_iou=0.5)
    _run(reg, [(0.0, {1: A, 2: A_JITTER, 3: B}), (10.0, {1: A, 2: A_JITTER, 3: B})])
    reg.resolve(10.5, {1: A, 3: B})                       # the duplicate vanishes
    assert 2 in reg.pop_cleared()
    reg.resolve(11.0, {3: B})                             # the student leaves
    assert reg.resolve(30.0, {7: A_JITTER, 3: B}) == {7: 1, 3: 3}
    assert reg.resolve(31.0, {7: A_JITTER, 3: B, 8: C})[8] == 2, \
        "the withdrawn number is the next one handed out"


def test_two_tracker_ids_swapping_students_do_not_swap_seat_numbers():
    """The tracker moved live ids between students on a real upload. The number
    must stay with the place, or the table shows two students trading cues."""
    reg = SeatRegistry()
    _run(reg, [(t, {1: A, 2: B}) for t in (0.0, 1.0, 2.0, 3.0)])
    assert reg.resolve(3.1, {1: B, 2: A}) == {1: 2, 2: 1}
    assert reg.pop_cleared() == [], "each seat's history is still its own student's"
    assert {e[3] for e in reg.events if e[0] == 3.1} >= {"jumped"}


def test_a_live_id_hopping_to_a_new_place_leaves_its_seat_behind():
    reg = SeatRegistry()
    _run(reg, [(t, {1: A, 2: B}) for t in (0.0, 1.0, 2.0, 3.0)])
    assert reg.resolve(3.1, {1: C, 2: B}) == {1: 3, 2: 2}
    assert reg.resolve(20.0, {9: A_JITTER, 2: B, 1: C})[9] == 1, \
        "the student who returns to A gets A's number back"


def test_a_slow_lean_moves_the_seat_with_the_student():
    reg = SeatRegistry()
    box = list(A)
    for i in range(60):                      # 3 px per step, 180 px in total
        box = [box[0] + 3, box[1], box[2] + 3, box[3]]
        assert reg.resolve(i * 0.05, {1: box}) == {1: 1}


def test_a_seat_carried_onto_an_established_seat_hands_back_its_number():
    """The 11-minute upload, reduced: a student held seat 1 for 30 s and left;
    someone opened seat 2 elsewhere and that seat was carried onto seat 1's spot.
    The spot's established number wins, and the live history moves with it."""
    reg = SeatRegistry()
    _run(reg, [(i * 0.5, {1: A}) for i in range(61)])          # seat 1 for 30 s
    reg.resolve(30.5, {})
    # Seat 2 opens at B well after seat 1 emptied, so this is not the
    # short-window fold, then is carried onto A and held there.
    _run(reg, [(40.0 + i * 0.5, {2: B}) for i in range(16)])
    box, t, seat = list(B), 48.0, None
    for step in range(120):                    # B -> A at 3 px a step, then hold
        if step < 100:
            box = [box[0] - 3, box[1], box[2] - 3, box[3]]
        t += 0.05
        seat = reg.resolve(t, {2: box})[2]
    assert seat == 1, "the spot's established number must win"
    assert (2, 1) in reg.pop_moved()


def test_a_large_seat_does_not_reach_a_small_box_far_from_it():
    """A passer-by's large box left a seat whose tolerance, measured on that
    seat alone, reached a seated student 275 px away."""
    reg = SeatRegistry()
    big = [100.0, -70.0, 740.0, 570.0]         # side 640; centre 275 px from A's
    _run(reg, [(i * 0.5, {1: big}) for i in range(10)])
    reg.resolve(5.0, {})
    assert reg.resolve(5.5, {2: A}) == {2: 2}


def test_the_closest_free_seat_is_taken():
    reg = SeatRegistry()
    near_a = [100.0, 100.0, 200.0, 290.0]
    _run(reg, [(0.0, {1: A, 2: near_a}), (3.0, {1: A, 2: near_a}), (3.5, {})])
    assert reg.resolve(4.0, {9: near_a}) == {9: 2}
