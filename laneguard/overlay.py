import cv2
import numpy as np

from .monitor import State


def render(frame, lanes, geometry, status, inference_ms, fps, provider):
    color = {State.NORMAL: (0, 210, 0), State.WARNING: (0, 0, 255),
             State.UNRELIABLE: (0, 190, 255)}[status.state]
    for index in (1, 2):
        points = np.asarray(lanes[index], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(frame, [points], False, (255, 200, 0), 2)
    if geometry:
        row = int(geometry.row)
        center = int((geometry.left + geometry.right) / 2)
        cv2.line(frame, (int(geometry.left), row), (int(geometry.right), row), color, 2)
        cv2.circle(frame, (center, row), 6, (0, 255, 255), -1)
        cv2.drawMarker(frame, (int(geometry.vehicle), row), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    offset = 'N/A' if status.offset is None else f'{status.offset:+.3f}'
    lines = [status.state.value,
             f'Offset {offset} (+RIGHT / -LEFT) | Pending {status.pending}',
             f'FPS {fps:.1f} | Inference {inference_ms:.1f} ms',
             f'{provider} | Q/ESC quit',
             'Research demo - image-relative estimate, not for driving']
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 148), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (10, 25 + index * 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color if index == 0 else (255, 255, 255), 1, cv2.LINE_AA)
    return frame
