from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import download_iitb_dataset as downloader
from src.data.cleaning import TranslationPair
from src.data.prepared import file_digest


def test_schema_requires_string_translation_fields(tmp_path: Path) -> None:
    assert downloader.convert_record({"translation": {"en": "Hello", "hi": "नमस्ते"}}) == TranslationPair("Hello", "नमस्ते")
    for row in ({"translation": {"en": "Hi"}}, {"translation": {"en": "Hi", "hi": None}}, {"en": "Hi", "hi": "नमस्ते"}):
        with pytest.raises(ValueError, match="translation"):
            downloader.convert_record(row)
    bad = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pylist([{"translation": {"en": "Hi", "hi": 1}}]), bad)
    with pq.ParquetFile(bad) as file, pytest.raises(ValueError, match="hi must be a string"):
        downloader.validate_schema(file, "train")


def test_reservoir_is_seeded_reaches_later_rows_and_rejects_oversampling() -> None:
    rows = [TranslationPair(str(index), str(index)) for index in range(100)]
    first, count = downloader.reservoir(iter(rows), 10, 42)
    assert count == 100 and first != rows[:10]
    assert first == downloader.reservoir(iter(rows), 10, 42)[0]
    assert first != downloader.reservoir(iter(rows), 10, 43)[0]
    with pytest.raises(ValueError, match="only 100"):
        downloader.reservoir(iter(rows), 101, 42)


def test_pinning_and_repeatable_acquisition_use_only_published_files(tmp_path, monkeypatch) -> None:
    sizes = {"train": 80, "validation": 4, "test": 5}
    for split, count in sizes.items():
        records = [{"translation": {"en": f"{split} {i}", "hi": f"वाक्य {i}"}} for i in range(count)]
        pq.write_table(pa.Table.from_pylist(records), tmp_path / f"{split}-00000-of-00001.parquet")
    calls = []
    monkeypatch.setattr(downloader, "HfApi", lambda: SimpleNamespace(dataset_info=lambda *args, **kwargs:
                        SimpleNamespace(sha=downloader.REVISION, gated=False, private=False)))

    def download(dataset, filename, **options):
        calls.append((dataset, filename, options))
        return tmp_path / Path(filename).name

    monkeypatch.setattr(downloader, "hf_hub_download", download)
    a, b = tmp_path / "a", tmp_path / "b"
    first = downloader.acquire(downloader.DATASET, "main", 13, 42, a, offline=False, force=False)
    second = downloader.acquire(downloader.DATASET, downloader.REVISION, 13, 42, b, offline=True, force=False)
    assert first["official_split_sizes"] == sizes
    assert first["files"] == second["files"]
    assert first["revision"] == downloader.REVISION
    assert first["license"] == downloader.LICENSE and first["license_warning"]
    assert first["selected_train_sha256"] == file_digest(a / downloader.FILES["train"])
    assert "validation" not in (a / downloader.FILES["train"]).read_text()
    assert len(calls) == 6
    assert all(options["revision"] == downloader.REVISION and options["token"] is False for _, _, options in calls)
    assert all(options["local_files_only"] for _, _, options in calls[3:])
    with pytest.raises(ValueError, match="exact"):
        downloader.resolve_revision(downloader.DATASET, "main", offline=True)
    with pytest.raises(ValueError, match="Output exists"):
        downloader.acquire(downloader.DATASET, downloader.REVISION, 13, 42, a, offline=True, force=False)
