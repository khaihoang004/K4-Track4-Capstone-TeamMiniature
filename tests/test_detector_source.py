from types import SimpleNamespace
import sys

import numpy as np
import pytest

from laneguard.detector import LaneDetector, ModelError, decode, preprocess
from laneguard.source import CameraSource, SourceError


def logits():
    values = np.full((1, 101, 56, 4), -20, dtype=np.float32)
    values[:, 100, :, :] = 20  # missing outer lanes
    values[:, 100, :, 1:3] = -20
    values[:, 25, :, 1] = 20
    values[:, 75, :, 2] = 20
    return values


def test_preprocessing_rgb_normalization():
    frame = np.zeros((20, 30, 3), np.uint8)
    frame[:, :, 2] = 255
    tensor = preprocess(frame)
    assert tensor.shape == (1, 3, 288, 800)
    assert tensor.dtype == np.float32
    assert tensor[0, :, 0, 0] == pytest.approx([(1-.485)/.229, -.456/.224, -.406/.225])


def test_decode_anchors_and_background():
    result = decode(logits(), 'tusimple', 1280, 720)
    assert len(result[1]) == 56 and len(result[2]) == 56
    assert result[0] == [] and result[3] == []
    assert result[1][0][1] == pytest.approx(720 * 284 / 288 - 1)
    assert result[1][0][0] == pytest.approx(26 * 799/99 * 1280/800 - 1)
    assert decode(np.full((101, 56, 4), np.nan), 'tusimple', 1000, 1000) == [[], [], [], []]


def test_wrong_output_rejected():
    with pytest.raises(ModelError):
        decode(np.zeros((1, 2, 3)), 'tusimple', 1000, 1000)


def test_missing_model(tmp_path):
    with pytest.raises(ModelError, match='Missing model'):
        LaneDetector(tmp_path / 'missing.onnx')


def fake_runtime(monkeypatch, shape=None, failure=False):
    class Session:
        def __init__(self, path, providers):
            if failure:
                raise RuntimeError('invalid protobuf')
            self.providers = providers

        def get_inputs(self):
            return [SimpleNamespace(name='input', shape=[1, 3, 288, 800], type='tensor(float)')]

        def get_outputs(self):
            return [SimpleNamespace(name='output', shape=shape or [1, 101, 56, 4])]

        def get_providers(self):
            return self.providers

        def run(self, names, inputs):
            return [logits()]

    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        InferenceSession=Session, get_available_providers=lambda: ['CPUExecutionProvider']))


@pytest.mark.parametrize('failure,shape', [(True, None), (False, [1, 5, 6, 7])])
def test_broken_or_incompatible_model(tmp_path, monkeypatch, failure, shape):
    model = tmp_path / 'model.onnx'
    model.touch()
    fake_runtime(monkeypatch, shape=shape, failure=failure)
    with pytest.raises(ModelError):
        LaneDetector(model)


def test_provider_fallback_and_mock_inference(tmp_path, monkeypatch):
    model = tmp_path / 'model.onnx'
    model.touch()
    fake_runtime(monkeypatch)
    detector = LaneDetector(model, provider='CUDAExecutionProvider')
    assert detector.providers == ['CPUExecutionProvider']
    lanes, ms = detector.detect(np.zeros((720, 1280, 3), np.uint8))
    assert len(lanes[1]) == 56 and ms >= 0


def test_camera_initialization_failure(monkeypatch):
    released = []
    monkeypatch.setattr('laneguard.source.cv2.VideoCapture', lambda _: SimpleNamespace(
        isOpened=lambda: False, release=lambda: released.append(True)))
    with pytest.raises(SourceError, match='Cannot open webcam'):
        CameraSource(0)
    assert released


def test_gpu_initialization_failure_retries_cpu(tmp_path, monkeypatch):
    model = tmp_path / 'model.onnx'
    model.touch()
    fake_runtime(monkeypatch)
    runtime = sys.modules['onnxruntime']
    original_session = runtime.InferenceSession
    calls = []

    def session(path, providers):
        calls.append(providers[0])
        if providers[0] == 'CUDAExecutionProvider':
            raise RuntimeError('CUDA shared library unavailable')
        return original_session(path, providers)

    runtime.get_available_providers = lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider']
    runtime.InferenceSession = session
    detector = LaneDetector(model, provider='CUDAExecutionProvider')
    assert detector.providers == ['CPUExecutionProvider']
    assert calls == ['CUDAExecutionProvider', 'CPUExecutionProvider']
