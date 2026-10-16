from io import BytesIO
import tarfile

import pytest

from tools.extract_model import MODEL_NAME, extract


def archive(path, name, payload=b'mock model'):
    with tarfile.open(path, 'w:gz') as stream:
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        stream.addfile(member, BytesIO(payload))


def test_extract_copies_basename_to_fixed_destination(tmp_path):
    source = tmp_path / 'models.tar.gz'
    archive(source, '../../outside/' + MODEL_NAME)
    target = tmp_path / 'models' / MODEL_NAME
    extract(source, target)
    assert target.read_bytes() == b'mock model'
    with pytest.raises(ValueError, match='overwrite'):
        extract(source, target)


def test_missing_model_and_existing_partial_preserved(tmp_path):
    source = tmp_path / 'models.tar.gz'
    target = tmp_path / MODEL_NAME
    archive(source, 'unrelated.file')
    with pytest.raises(ValueError, match='not found'):
        extract(source, target)
    assert not target.exists()
    archive(source, MODEL_NAME)
    partial = target.with_suffix('.onnx.part')
    partial.write_bytes(b'existing download')
    with pytest.raises(FileExistsError):
        extract(source, target)
    assert partial.read_bytes() == b'existing download'
