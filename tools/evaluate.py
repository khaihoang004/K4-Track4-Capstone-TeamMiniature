"""Evaluate manually labeled source frames against per-clip LaneGuard logs."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def evaluate(labels_path, prediction_paths):
    predictions = {}
    scenarios = set()
    latency, inference, fps = [], [], []
    for path in prediction_paths:
        clip = Path(path).stem
        with open(path, newline='') as stream:
            for row in csv.DictReader(stream):
                key = (clip, int(row['source_frame']))
                if key in predictions:
                    raise ValueError(f'Duplicate prediction: {key}')
                if row['state'] not in ('NORMAL', 'LANE DEPARTURE WARNING', 'LANE NOT RELIABLE'):
                    raise ValueError(f'Unknown prediction state: {row["state"]}')
                predictions[key] = row['state']
                latency.append(float(row['latency_ms']))
                inference.append(float(row['inference_ms']))
                if float(row['fps']) > 0:
                    fps.append(float(row['fps']))
    tp = fp = tn = fn = unreliable = 0
    seen = set()
    with open(labels_path, newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['clip'], int(row['source_frame']))
            if key in seen:
                raise ValueError(f'Duplicate label: {key}')
            seen.add(key)
            if key not in predictions:
                raise ValueError(f'No prediction for labeled frame {key}')
            if row['label'] not in ('normal', 'warning'):
                raise ValueError('Labels must be normal or warning; omit ambiguous samples')
            scenarios.add(row['clip'])
            state = predictions[key]
            unreliable += state == 'LANE NOT RELIABLE'
            positive = row['label'] == 'warning'
            warning = state == 'LANE DEPARTURE WARNING'
            tp += positive and warning
            fp += not positive and warning
            tn += not positive and not warning
            fn += positive and not warning
    return dict(clips=len(scenarios), labeled_samples=len(seen),
                warning_samples=tp + fn, normal_samples=tn + fp,
                true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn,
                warning_precision=tp / (tp + fp) if tp + fp else None,
                warning_recall=tp / (tp + fn) if tp + fn else None,
                unreliable_samples=unreliable,
                reliable_coverage=(len(seen) - unreliable) / len(seen) if seen else None,
                mean_processing_ms=mean(latency) if latency else None,
                mean_inference_ms=mean(inference) if inference else None,
                mean_rolling_fps=mean(fps) if fps else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--predictions', nargs='+', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.labels, args.predictions), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Evaluation error: {exc}\n')
