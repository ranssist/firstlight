"""입력 소스 처리 — 이미지 시퀀스와 동영상.

타임스탬프가 **실제 초**여야 한다는 것이 핵심이다. 프레임 번호를 시각으로
쓰면 성장률(1/s)이 프레임률에 따라 달라져 dense/sparse 사이에서 특징이
전혀 다른 의미가 된다.
"""

import cv2
import numpy as np
import pytest

from firstlight.cli import _open_source


def write_video(path, n_frames=30, fps=25.0, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), (i * 7) % 255, np.uint8))
    writer.release()
    return path


def write_images(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        cv2.imwrite(str(directory / name), np.zeros((64, 64, 3), np.uint8))
    return directory


# ------------------------------------------------------------------ 동영상


def test_video_interval_comes_from_fps(tmp_path):
    reader = _open_source(str(write_video(tmp_path / "a.mp4", fps=25.0)))
    assert reader.interval_s == pytest.approx(1 / 25, rel=0.02)


def test_video_yields_frames_with_second_timestamps(tmp_path):
    reader = _open_source(str(write_video(tmp_path / "a.mp4", n_frames=10, fps=20.0)))
    frames = list(reader)
    assert len(frames) == 10
    assert frames[0][1] == pytest.approx(0.0)
    assert frames[4][1] == pytest.approx(4 / 20, rel=0.02)


def test_video_respects_max_frames(tmp_path):
    reader = _open_source(str(write_video(tmp_path / "a.mp4", n_frames=30)), max_frames=7)
    assert len(list(reader)) == 7


# -------------------------------------------------------------- 이미지 시퀀스


def test_figlib_offsets_are_parsed_as_seconds(tmp_path):
    """FIgLib 파일명의 ±초 오프셋을 그대로 타임스탬프로 써야 한다."""
    directory = write_images(
        tmp_path / "seq",
        ["1465063200_-02400.jpg", "1465063260_-02340.jpg", "1465067760_+02160.jpg"],
    )
    reader = _open_source(str(directory))
    stamps = [t for _, t in reader]
    assert stamps == [-2400.0, -2340.0, 2160.0]
    assert reader.interval_s == pytest.approx(60.0)


def test_frames_are_sorted_by_time(tmp_path):
    directory = write_images(
        tmp_path / "seq",
        ["a_+00120.jpg", "b_-00060.jpg", "c_+00000.jpg"],
    )
    stamps = [t for _, t in _open_source(str(directory))]
    assert stamps == sorted(stamps)


def test_plain_filenames_fall_back_to_one_second_steps(tmp_path):
    directory = write_images(tmp_path / "seq", ["a.jpg", "b.jpg", "c.jpg"])
    reader = _open_source(str(directory))
    assert [t for _, t in reader] == [0.0, 1.0, 2.0]
    assert reader.interval_s == pytest.approx(1.0)


def test_non_image_files_are_ignored(tmp_path):
    directory = write_images(tmp_path / "seq", ["a.jpg", "b.jpg"])
    (directory / "notes.txt").write_text("무시돼야 한다", encoding="utf-8")
    assert len(list(_open_source(str(directory)))) == 2


# --------------------------------------------------------------------- 오류


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _open_source(str(tmp_path / "없는경로"))


def test_empty_directory_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="이미지가 없다"):
        _open_source(str(tmp_path / "empty"))


def test_mode_follows_source_rate(tmp_path):
    """동영상은 dense, FIgLib 시퀀스는 sparse 로 이어져야 한다."""
    from firstlight.verify.features import VerifierMode

    video = _open_source(str(write_video(tmp_path / "a.mp4", fps=25.0)))
    assert VerifierMode.from_interval(video.interval_s) is VerifierMode.DENSE

    directory = write_images(
        tmp_path / "seq", ["a_-00120.jpg", "b_-00060.jpg", "c_+00000.jpg"]
    )
    images = _open_source(str(directory))
    assert VerifierMode.from_interval(images.interval_s) is VerifierMode.SPARSE
