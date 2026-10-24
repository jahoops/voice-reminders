#!/usr/bin/env python3
"""Archive and transcribe new VN-541PC voice memos."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


DEFAULT_SOURCE = Path("/media/j/VN_541PC/RECORDER/MEMO")
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "captures"
FILENAME_DATE = re.compile(r"^(\d{2})(\d{2})(\d{2})_")


def iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def date_from_filename(filename: str) -> str | None:
    match = FILENAME_DATE.match(filename)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%y%m%d").date().isoformat()
    except ValueError:
        return None


def load_capture_ids(log_path: Path) -> set[str]:
    ids: set[str] = set()
    if not log_path.exists():
        return ids
    with log_path.open("r", encoding="utf-8") as log:
        for line_number, line in enumerate(log, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                capture_id = record["capture_id"]
                if isinstance(capture_id, str):
                    ids.add(capture_id)
            except (json.JSONDecodeError, KeyError, TypeError):
                print(
                    f"warning: ignoring incomplete/invalid log line {line_number}",
                    file=sys.stderr,
                )
    return ids


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_source(source: Path, archive_dir: Path, capture_id: str) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / f"{source.stem}-{capture_id[:12]}{source.suffix}"

    if destination.exists():
        if not destination.is_file() or hash_file(destination) != capture_id:
            raise RuntimeError(f"archive collision or damaged archive: {destination}")
        return destination

    before = source.stat()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=archive_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())

        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise RuntimeError("source changed while it was being copied")
        if hash_file(temporary) != capture_id:
            raise RuntimeError("copied audio failed SHA-256 verification")

        # The batch lock prevents another processor from publishing this path.
        # Refuse an unexpected pre-existing destination rather than overwriting it.
        if destination.exists():
            if hash_file(destination) != capture_id:
                raise RuntimeError(f"archive collision: {destination}")
        else:
            os.rename(temporary, destination)
            directory_fd = os.open(archive_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def append_record(log_path: Path, record: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with log_path.open("a+b", buffering=0) as log:
        log.seek(0, os.SEEK_END)
        if log.tell():
            log.seek(-1, os.SEEK_END)
            if log.read(1) != b"\n":
                log.seek(0, os.SEEK_END)
                log.write(b"\n")
        log.seek(0, os.SEEK_END)
        log.write(encoded.encode("utf-8"))
        os.fsync(log.fileno())


def whisper_transcriber() -> Callable[[Path], str]:
    from faster_whisper import WhisperModel

    model = WhisperModel("small.en", device="cpu", compute_type="int8")

    def transcribe(path: Path) -> str:
        segments, _ = model.transcribe(str(path))
        return " ".join(segment.text.strip() for segment in segments).strip()

    return transcribe


def process_batch(
    source_dir: Path,
    data_dir: Path,
    transcriber_factory: Callable[[], Callable[[Path], str]] = whisper_transcriber,
) -> tuple[int, int, int]:
    audio_dir = data_dir / "audio"
    log_path = data_dir / "captures.jsonl"
    files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wma")
    captured = load_capture_ids(log_path)
    pending: list[tuple[Path, str]] = []
    queued_ids: set[str] = set()
    skipped = failed = 0

    for source in files:
        try:
            capture_id = hash_file(source)
            if capture_id in captured or capture_id in queued_ids:
                skipped += 1
            else:
                pending.append((source, capture_id))
                queued_ids.add(capture_id)
        except Exception as error:
            failed += 1
            print(f"failed {source.name}: {error}", file=sys.stderr)

    transcribe = transcriber_factory() if pending else None
    processed = 0
    for source, capture_id in pending:
        try:
            source_stat = source.stat()
            archived = archive_source(source, audio_dir, capture_id)
            assert transcribe is not None
            transcription = transcribe(archived)
            record = {
                "capture_id": capture_id,
                "source_filename": source.name,
                "original_source_path": str(source.resolve()),
                "archived_audio_path": str(archived.resolve()),
                "filename_date": date_from_filename(source.name),
                "source_mtime_observed": iso_timestamp(source_stat.st_mtime),
                "imported_at": datetime.now(timezone.utc).isoformat(),
                "transcription": transcription,
            }
            append_record(log_path, record)
            captured.add(capture_id)
            processed += 1
        except Exception as error:
            failed += 1
            print(f"failed {source.name}: {error}", file=sys.stderr)
    return processed, skipped, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    if not args.source.is_dir():
        parser.error(f"recorder memo directory not found: {args.source}")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.data_dir / ".processor.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another voice-reminders processor is already running", file=sys.stderr)
            return 2
        processed, skipped, failed = process_batch(args.source, args.data_dir)

    print(f"processed={processed} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
