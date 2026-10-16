"""Copy only the verified TuSimple ONNX basename from PINTO's multi-format archive."""

import argparse
from pathlib import Path
import shutil
import tarfile


MODEL_NAME = 'ultra_falst_lane_detection_tusimple_288x800.onnx'


def extract(archive_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.onnx.part')
    created = False
    try:
        with tarfile.open(archive_path, 'r|gz') as archive:
            for member in archive:
                if member.isfile() and Path(member.name).name == MODEL_NAME:
                    # No archive paths are written; ignore links and arbitrary directory entries.
                    if member.size > 1024**3:
                        raise ValueError('Unexpected model size (> 1 GiB)')
                    with temporary.open('xb') as target, archive.extractfile(member) as source:
                        created = True
                        shutil.copyfileobj(source, target)
                    temporary.replace(output)
                    return
        raise ValueError(f'{MODEL_NAME} not found in archive')
    finally:
        if created:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--output', type=Path, default=Path('models') / MODEL_NAME)
    args = parser.parse_args()
    try:
        extract(args.archive, args.output)
        print(f'Model saved to {args.output}; run the image smoke check in README.md')
    except (OSError, ValueError, EOFError, tarfile.TarError) as exc:
        parser.exit(1, f'Extraction error: {exc}\n')
