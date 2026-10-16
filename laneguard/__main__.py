"""Run with python -m laneguard --help."""

import argparse
from collections import deque
import csv
import json
import logging
from pathlib import Path
from time import perf_counter

try:
    import cv2
    from .detector import LaneDetector, ModelError
    from .geometry import estimate_geometry
    from .monitor import DepartureMonitor
    from .overlay import render
    from .source import CameraSource, SourceError
except ImportError as exc:
    raise SystemExit(f'Missing dependency ({exc}). Run: python -m pip install -r requirements.txt') from exc


def parser():
    p = argparse.ArgumentParser(description="LaneGuard UFLD v1 research demo")
    p.add_argument('--model', default='models/ultra_falst_lane_detection_tusimple_288x800.onnx')
    p.add_argument('--dataset', choices=['tusimple', 'culane'], default='tusimple')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--camera', type=int, default=0)
    source.add_argument('--video', type=Path)
    source.add_argument('--image', type=Path, help='Single-frame inference smoke check')
    p.add_argument('--provider', default='CPUExecutionProvider')
    p.add_argument('--threshold', type=float, default=0.65)
    p.add_argument('--clear-threshold', type=float, default=0.50)
    p.add_argument('--persistence', type=int, default=5)
    p.add_argument('--alpha', type=float, default=0.35)
    p.add_argument('--lookahead', type=float, default=0.80)
    p.add_argument('--reference', type=float, default=0.50)
    p.add_argument('--min-width', type=float, default=0.15)
    p.add_argument('--max-width', type=float, default=0.90)
    p.add_argument('--min-lane-probability', type=float, default=0.50)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--max-frames', type=int, default=0, help='0 = until EOF or quit')
    p.add_argument('--log', type=Path, help='Per-frame CSV evidence')
    p.add_argument('--output', type=Path, help='Save last annotated frame (e.g. runs/frame.jpg)')
    return p


def run(args):
    if not (0 < args.lookahead < 1 and 0 <= args.reference <= 1
            and 0 < args.min_width < args.max_width <= 1
            and 0 <= args.min_lane_probability <= 1 and args.max_frames >= 0):
        raise ValueError('Invalid geometry, probability, or frame-limit configuration')
    monitor = DepartureMonitor(args.threshold, args.clear_threshold, args.persistence, args.alpha)
    detector = LaneDetector(args.model, args.dataset, args.provider, args.min_lane_probability)
    source = None
    log_file = None
    count = 0
    inference_sum = latency_sum = 0.0
    timestamps = deque()
    started = perf_counter()
    annotated = None
    try:
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise SourceError(f'Cannot decode image: {args.image}')
        else:
            source = CameraSource(str(args.video) if args.video else args.camera)
        writer = None
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            metadata = dict(configuration={k: str(v) if isinstance(v, Path) else v
                                            for k, v in vars(args).items()},
                            providers=detector.providers, opencv_version=cv2.__version__)
            args.log.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            log_file = args.log.open('w', newline='')
            writer = csv.DictWriter(log_file, fieldnames=[
                'frame', 'source_frame', 'elapsed_s', 'state', 'raw_offset', 'offset',
                'inference_ms', 'latency_ms', 'frame_age_ms', 'fps', 'provider'])
            writer.writeheader()
        started = perf_counter()
        while True:
            frame_started = perf_counter()
            item = (frame, frame_started, 1) if args.image else source.read()
            if item is None:
                break
            frame, acquired, source_frame = item
            lanes, inference_ms = detector.detect(frame)
            h, w = frame.shape[:2]
            geometry = estimate_geometry(lanes, w, h, args.lookahead, args.reference,
                                         args.min_width, args.max_width)
            status = monitor.update(geometry)
            now = perf_counter()
            timestamps.append(now)
            while len(timestamps) > 2 and now - timestamps[0] > 2:
                timestamps.popleft()
            fps = (len(timestamps) - 1) / (now - timestamps[0]) if len(timestamps) > 1 else 0.0
            annotated = render(frame, lanes, geometry, status, inference_ms, fps, detector.providers[0])
            finished = perf_counter()
            latency_ms = (finished - frame_started) * 1000
            age_ms = (finished - acquired) * 1000
            cv2.putText(annotated, f'Processing {latency_ms:.1f} ms | Frame age {age_ms:.1f} ms',
                        (10, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            count += 1
            inference_sum += inference_ms
            latency_sum += latency_ms
            if writer:
                writer.writerow(dict(frame=count, source_frame=source_frame,
                                     elapsed_s=finished - started, state=status.state.value,
                                     raw_offset=geometry.offset if geometry else '',
                                     offset=status.offset if status.offset is not None else '',
                                     inference_ms=inference_ms, latency_ms=latency_ms,
                                     frame_age_ms=age_ms, fps=fps, provider=';'.join(detector.providers)))
            if not args.headless:
                cv2.imshow('LaneGuard AI', annotated)
                if cv2.waitKey(0 if args.image else 1) & 0xFF in (27, ord('q'), ord('Q')):
                    break
            if args.image or (args.max_frames and count >= args.max_frames):
                break
        if args.output and annotated is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.output), annotated):
                raise SourceError(f'Cannot write image: {args.output}')
        elapsed = perf_counter() - started
        return dict(frames=count, elapsed_s=elapsed,
                    average_fps=count / elapsed if elapsed else 0,
                    mean_inference_ms=inference_sum / count if count else None,
                    mean_latency_ms=latency_sum / count if count else None,
                    providers=detector.providers)
    finally:
        if source:
            source.close()
        if log_file:
            log_file.close()
        if not args.headless:
            cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parser().parse_args()
    try:
        print(json.dumps(run(args), indent=2))
        return 0
    except (ModelError, SourceError, ValueError, OSError, cv2.error) as exc:
        logging.error('%s', exc)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
