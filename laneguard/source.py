"""File input is sequential; webcam input keeps only the newest frame."""

from threading import Condition, Event, Thread
from time import perf_counter

import cv2


class SourceError(RuntimeError):
    pass


class CameraSource:
    def __init__(self, source=0):
        self.live = isinstance(source, int)
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise SourceError(f"Cannot open {'webcam' if self.live else 'video'} {source!r}; check device/path and permissions")
        self.stop = Event()
        self.condition = Condition()
        self.latest = None
        self.sequence = 0
        self.consumed = 0
        self.error = None
        self.thread = None
        self.file_frames = 0
        if self.live:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; not every backend supports it.
            self.thread = Thread(target=self._capture, daemon=True)
            self.thread.start()

    def _capture(self):
        failures = 0
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                acquired = perf_counter()
                if not ok:
                    failures += 1
                    if failures >= 20:
                        raise SourceError("Webcam stopped returning frames after 20 retries")
                    self.stop.wait(0.05)
                    continue
                failures = 0
                with self.condition:
                    self.sequence += 1
                    self.latest = (frame, acquired, self.sequence)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()
        finally:
            self.capture.release()

    def read(self):
        if not self.live:
            ok, frame = self.capture.read()
            if not ok:
                if self.file_frames == 0:
                    raise SourceError("Video opened but no frame could be decoded")
                return None
            self.file_frames += 1
            return frame, perf_counter(), self.file_frames
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence > self.consumed or self.error, timeout=3)
            if self.error:
                raise SourceError(self.error)
            if not ready:
                raise SourceError("Webcam timed out waiting for a new frame")
            self.consumed = self.sequence
            return self.latest

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
            # The worker releases after its read returns; avoid concurrent release/read.
        else:
            self.capture.release()
