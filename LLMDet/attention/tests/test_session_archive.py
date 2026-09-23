"""Analysed recordings must survive a Space restart.

On the Space, ``uploads/`` and ``sessions/`` are on the container's ephemeral
disk, so every recording analysed since boot vanished on the next restart or
sleep, while the demo session fetched from the artifact repo came back. The
archive keeps a copy on the durable mount and restores it at boot.

What these pin is the part that is silent when wrong: a save that was cut off
must not come back as a truncated session that looks usable, a restored session
must be byte-identical, and nothing read back from storage may write outside
the session directory.
"""
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "dashboard"))

np = pytest.importorskip("numpy")
SRC = pytest.importorskip("sources")


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    up, ses, arc = tmp_path / "uploads", tmp_path / "sessions", tmp_path / "archive"
    up.mkdir()
    ses.mkdir()
    monkeypatch.setattr(SRC, "UPLOAD_DIR", up)
    monkeypatch.setattr(SRC, "SESSION_DIR", ses)
    monkeypatch.setattr(SRC, "ARCHIVE_DIR", arc)
    return up, ses, arc


def make_session(ses, up, name="lecture"):
    video = up / f"{name}.mp4"
    video.write_bytes(b"not really a video" * 100)
    d = ses / name
    (d / "frames").mkdir(parents=True)
    for i in range(5):
        (d / "frames" / f"{i:06d}.jpg").write_bytes(bytes([i]) * 1000)
    np.savez_compressed(d / "features.npz", frame_idx=np.arange(5))
    (d / "meta.json").write_text(json.dumps(
        {"n_frames": 5, "n_tracks": 2, "video": str(video)}))
    return d, video


def snapshot(d):
    return {p.relative_to(d).as_posix(): p.read_bytes()
            for p in sorted(d.rglob("*")) if p.is_file()}


def wipe(up, ses):
    """What a restart does to the ephemeral disk."""
    import shutil
    shutil.rmtree(up)
    shutil.rmtree(ses)
    up.mkdir()
    ses.mkdir()


def test_round_trip_is_byte_identical(dirs):
    up, ses, arc = dirs
    d, video = make_session(ses, up)
    before, video_bytes = snapshot(d), video.read_bytes()

    SRC.archive_session(d, video)
    wipe(up, ses)
    got = SRC.restore_archive()

    assert got["sessions"] == ["lecture"] and got["videos"] == ["lecture.mp4"]
    assert snapshot(ses / "lecture") == before
    assert (up / "lecture.mp4").read_bytes() == video_bytes


def test_restored_recording_lists_as_analysed_and_saved(dirs):
    up, ses, arc = dirs
    d, video = make_session(ses, up)
    SRC.archive_session(d, video)
    wipe(up, ses)
    SRC.restore_archive()

    by_kind = {s["kind"]: s for s in SRC.list_sources()}
    assert by_kind["session"]["name"] == "lecture" and by_kind["session"]["saved"]
    v = by_kind["video"]
    assert v["ready"] and v["saved"] and v["origin"] == "upload"


def test_meta_json_is_the_last_member(dirs):
    """A restore that stops early must leave nothing session_meta accepts."""
    up, ses, arc = dirs
    d, video = make_session(ses, up)
    tar_path = SRC.archive_session(d, video)
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
    assert names[-1] == "lecture/meta.json"
    assert names[-2] == "lecture/features.npz"


def test_interrupted_save_is_not_restored(dirs):
    up, ses, arc = dirs
    d, video = make_session(ses, up)
    tar_path = SRC.archive_session(d, video)
    SRC._done(tar_path).unlink()           # the process died before the marker
    wipe(up, ses)

    got = SRC.restore_archive()
    assert got["sessions"] == []
    assert ("lecture.tar", "save never finished") in got["skipped"]
    assert not (ses / "lecture").exists()


def test_existing_session_is_left_alone(dirs):
    up, ses, arc = dirs
    d, video = make_session(ses, up)
    SRC.archive_session(d, video)
    (d / "meta.json").write_text(json.dumps({"n_frames": 99, "n_tracks": 1}))

    assert SRC.restore_archive()["sessions"] == []
    assert json.loads((d / "meta.json").read_text())["n_frames"] == 99


def test_a_second_restore_does_nothing(dirs):
    up, ses, arc = dirs
    d, video = make_session(ses, up)
    SRC.archive_session(d, video)
    wipe(up, ses)
    SRC.restore_archive()
    got = SRC.restore_archive()
    assert got["sessions"] == [] and got["videos"] == []


def test_member_outside_the_session_is_refused(dirs, tmp_path):
    up, ses, arc = dirs
    (arc / "sessions").mkdir(parents=True)
    evil = arc / "sessions" / "lecture.tar"
    with tarfile.open(evil, "w") as tf:
        data = b"pwned"
        info = tarfile.TarInfo("lecture/../../escaped.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    SRC._done(evil).write_text("")

    got = SRC.restore_archive()
    assert got["sessions"] == []
    assert not (tmp_path / "escaped.txt").exists()
    assert not any(ses.iterdir()), "a refused restore must leave no staging dir"


def test_half_unpacked_restore_is_not_listed(dirs):
    up, ses, arc = dirs
    d, _ = make_session(ses, up)
    d.rename(ses / ".restoring-lecture")
    assert [s for s in SRC.list_sources() if s["kind"] == "session"] == []


def test_no_archive_configured_keeps_nothing(dirs, monkeypatch):
    up, ses, arc = dirs
    monkeypatch.setattr(SRC, "ARCHIVE_DIR", None)
    d, video = make_session(ses, up)
    assert SRC.archive_session(d, video) is None
    assert SRC.restore_archive() == {"sessions": [], "videos": [], "skipped": []}
    assert not arc.exists()


def test_unfinished_session_is_not_archived(dirs):
    up, ses, arc = dirs
    d, video = make_session(ses, up)
    (d / "meta.json").unlink()
    with pytest.raises(ValueError):
        SRC.archive_session(d, video)
