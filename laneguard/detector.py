"""UFLD v1 ONNX adapter. See docs/model.md for verified tensor contract.

Pre/postprocessing adapted from ibaiGorordo's MIT-licensed reference;
ImageNet RGB normalization, reversed row anchors and softmax grid expectation.
"""

from pathlib import Path
from time import perf_counter
import logging

import cv2
import numpy as np


class ModelError(RuntimeError):
    pass


ANCHORS = {
    "tusimple": list(range(64, 285, 4)),
    "culane": [121, 131, 141, 150, 160, 170, 180, 189, 199,
               209, 219, 228, 238, 248, 258, 267, 277, 287],
}
GRIDS = {"tusimple": 100, "culane": 200}


def preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (800, 288)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(((resized - mean) / std).transpose(2, 0, 1)[None])


def decode(output, dataset, width, height, min_probability=0.5):
    grids = GRIDS[dataset]
    anchors = ANCHORS[dataset]
    values = np.asarray(output)
    if values.shape == (1, grids + 1, len(anchors), 4):
        values = values[0]
    if values.shape != (grids + 1, len(anchors), 4):
        raise ModelError(f"Unexpected output {values.shape}; expected UFLD v1 {dataset} logits")
    if not np.isfinite(values).all():
        return [[], [], [], []]
    values = values[:, ::-1, :]
    logits = values[:-1]
    weights = np.exp(logits - logits.max(axis=0, keepdims=True))
    weights /= weights.sum(axis=0, keepdims=True)
    locations = (weights * np.arange(1, grids + 1)[:, None, None]).sum(axis=0)
    # Extra uncertainty gate: background probability across ALL grid classes.
    all_weights = np.exp(values - values.max(axis=0, keepdims=True))
    lane_probability = 1 - all_weights[-1] / all_weights.sum(axis=0)
    valid = (values.argmax(axis=0) != grids) & (lane_probability >= min_probability)
    lanes = []
    for lane in range(4):
        points = []
        if valid[:, lane].sum() >= 3:
            for row in np.flatnonzero(valid[:, lane]):
                # Reference maps into dataset resolution; map directly to camera resolution.
                x = locations[row, lane] * (799 / (grids - 1)) * width / 800 - 1
                y = height * anchors[-1 - row] / 288 - 1
                if 0 <= x < width and 0 <= y < height:
                    points.append((float(x), float(y)))
        lanes.append(points)
    return lanes


class LaneDetector:
    def __init__(self, model_path, dataset="tusimple", provider="CPUExecutionProvider",
                 min_probability=0.5):
        if not Path(model_path).is_file():
            raise ModelError(f"Missing model: {model_path}. Follow docs/model.md to obtain UFLD v1 weights.")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelError("ONNX Runtime missing; install requirements.txt in your virtual environment") from exc
        self.dataset = dataset
        self.min_probability = min_probability
        available = ort.get_available_providers()
        if provider not in available:
            logging.warning("Provider %s unavailable (%s); falling back to CPU", provider, available)
            provider = "CPUExecutionProvider"
        if provider not in available:
            raise ModelError(f"No usable execution provider: {available}")
        providers = [provider]
        if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        try:
            try:
                self.session = ort.InferenceSession(str(model_path), providers=providers)
            except Exception as exc:
                if provider == 'CPUExecutionProvider' or 'CPUExecutionProvider' not in available:
                    raise
                logging.warning('Provider initialization failed (%s); retrying with CPU', exc)
                self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ModelError("Expected one input and one output; UFLD v2 is not supported")
            self.input, self.output = inputs[0], outputs[0]
            expected = [1, GRIDS[dataset] + 1, len(ANCHORS[dataset]), 4]
            output_shape = self.output.shape
            if output_shape != expected and output_shape != expected[1:]:
                raise ModelError(f"Incompatible output shape {output_shape}; expected {expected} (or unbatched)")
            if self.input.shape != [1, 3, 288, 800] or self.input.type != "tensor(float)":
                raise ModelError(f"Incompatible input: {self.input.shape} {self.input.type}; expected float32 [1,3,288,800]")
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Cannot load ONNX model: {exc}") from exc
        self.providers = self.session.get_providers()
        logging.info("Requested provider: %s; session providers: %s", provider, self.providers)

    def detect(self, frame):
        tensor = preprocess(frame)
        started = perf_counter()
        try:
            result = self.session.run([self.output.name], {self.input.name: tensor})[0]
        except Exception as exc:
            raise ModelError(f"ONNX inference failed: {exc}") from exc
        latency_ms = (perf_counter() - started) * 1000
        h, w = frame.shape[:2]
        return decode(result, self.dataset, w, h, self.min_probability), latency_ms
