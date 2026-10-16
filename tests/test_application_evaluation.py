import csv
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from laneguard.__main__ import parser, run
from laneguard.source import CameraSource, SourceError
from tools.evaluate import evaluate


def test_headless_video_pipeline_and_evaluation(tmp_path, monkeypatch):
    """Real video decode/render/log, mocked lane detections; no pretrained benchmark."""
    video = tmp_path / 'synthetic.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.zeros((480, 640, 3), np.uint8))
    writer.release()

    class Detector:
        providers = ['MOCK']

        def __init__(self, *args):
            self.index = 0

        def detect(self, frame):
            self.index += 1
            if self.index == 6:
                return [[], [], [], []], 0.0
            left, right = (120, 520) if self.index in (1, 7, 8) else (20, 350)
            return [[], [(left, y) for y in (340, 380, 420)],
                    [(right, y) for y in (340, 380, 420)], []], 0.0

    monkeypatch.setattr('laneguard.__main__.LaneDetector', Detector)
    log = tmp_path / 'synthetic.csv'
    output = tmp_path / 'last.jpg'
    args = parser().parse_args(['--video', str(video), '--headless', '--alpha', '1',
                                '--persistence', '3', '--log', str(log), '--output', str(output)])
    summary = run(args)
    assert summary['frames'] == 8
    assert output.is_file()
    assert log.with_suffix('.metadata.json').is_file()
    with log.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r['state'] for r in rows] == [
        'NORMAL', 'NORMAL', 'NORMAL', 'LANE DEPARTURE WARNING',
        'LANE DEPARTURE WARNING', 'LANE NOT RELIABLE', 'NORMAL', 'NORMAL']
    assert all(float(r['latency_ms']) >= 0 for r in rows)
    labels = tmp_path / 'labels.csv'
    labels.write_text('clip,source_frame,label\nsynthetic,1,normal\nsynthetic,4,warning\n'
                      'synthetic,6,warning\nsynthetic,8,warning\nsynthetic,5,normal\n')
    metrics = evaluate(labels, [log])
    assert metrics['true_positives'] == 1
    assert metrics['false_positives'] == 1
    assert metrics['true_negatives'] == 1
    assert metrics['false_negatives'] == 2
    assert metrics['warning_precision'] == pytest.approx(0.5)
    assert metrics['warning_recall'] == pytest.approx(1/3)
    assert metrics['reliable_coverage'] == pytest.approx(4/5)
    labels.write_text('clip,source_frame,label\nsynthetic,99,normal\n')
    with pytest.raises(ValueError, match='No prediction'):
        evaluate(labels, [log])


def test_video_opened_but_no_frames(monkeypatch):
    released = []
    monkeypatch.setattr('laneguard.source.cv2.VideoCapture', lambda _: SimpleNamespace(
        isOpened=lambda: True, read=lambda: (False, None), release=lambda: released.append(True)))
    source = CameraSource('broken.mp4')
    try:
        with pytest.raises(SourceError, match='no frame'):
            source.read()
    finally:
        source.close()
    assert released


def test_camera_latest_frame_and_cleanup(monkeypatch):
    from threading import Event
    released = Event()
    calls = []

    def read():
        calls.append(True)
        if len(calls) > 3:
            released.wait(0.02)
        return True, np.zeros((20, 20, 3), np.uint8)

    monkeypatch.setattr('laneguard.source.cv2.VideoCapture', lambda _: SimpleNamespace(
        isOpened=lambda: True, set=lambda *a: True, read=read, release=released.set))
    source = CameraSource(0)
    try:
        _, _, sequence = source.read()
        assert sequence >= 3  # Older frames replaced, no queue.
    finally:
        source.close()
    assert released.is_set()
