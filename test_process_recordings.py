import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from process_recordings import append_record, date_from_filename, process_batch


class ProcessorTests(unittest.TestCase):
    def test_filename_date(self):
        self.assertEqual(date_from_filename("260828_0003.WMA"), "2026-08-28")
        self.assertIsNone(date_from_filename("261332_0003.WMA"))
        self.assertIsNone(date_from_filename("note.WMA"))

    def test_process_retry_and_bad_file_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            data = root / "data"
            source.mkdir()
            good = source / "260828_0003.WMA"
            bad = source / "260828_0004.WMA"
            good.write_bytes(b"original audio")
            bad.write_bytes(b"bad audio")

            calls = []

            def factory():
                def transcribe(path):
                    calls.append(path)
                    if path.name.startswith(bad.stem):
                        raise RuntimeError("cannot transcribe")
                    return "remember this"
                return transcribe

            self.assertEqual(process_batch(source, data, factory), (1, 0, 1))
            records = [json.loads(line) for line in (data / "captures.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["capture_id"], hashlib.sha256(b"original audio").hexdigest())
            self.assertEqual(records[0]["transcription"], "remember this")
            archived = Path(records[0]["archived_audio_path"])
            self.assertEqual(archived.read_bytes(), b"original audio")
            self.assertIn(good.stem, archived.name)

            self.assertEqual(process_batch(source, data, factory), (0, 1, 1))
            self.assertEqual(len((data / "captures.jsonl").read_text().splitlines()), 1)

    def test_archived_without_log_entry_is_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            data = root / "data"
            source.mkdir()
            memo = source / "260828_0003.WMA"
            memo.write_bytes(b"audio")
            capture_id = hashlib.sha256(b"audio").hexdigest()
            audio_dir = data / "audio"
            audio_dir.mkdir(parents=True)
            archived = audio_dir / f"{memo.stem}-{capture_id[:12]}.WMA"
            archived.write_bytes(b"audio")

            self.assertEqual(process_batch(source, data, lambda: lambda path: "retried"), (1, 0, 0))
            record = json.loads((data / "captures.jsonl").read_text())
            self.assertEqual(record["transcription"], "retried")

    def test_same_bytes_are_captured_once_within_one_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "260828_0001.WMA").write_bytes(b"same")
            (source / "260828_0002.WMA").write_bytes(b"same")

            result = process_batch(source, root / "data", lambda: lambda path: "text")
            self.assertEqual(result, (1, 1, 0))
            self.assertEqual(len((root / "data" / "captures.jsonl").read_text().splitlines()), 1)

    def test_append_after_incomplete_tail_keeps_new_record_parseable(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "captures.jsonl"
            log.write_bytes(b'{"capture_id":"interrupted"')
            append_record(log, {"capture_id": "complete", "transcription": "ok"})
            lines = log.read_text().splitlines()
            self.assertEqual(json.loads(lines[-1])["capture_id"], "complete")


if __name__ == "__main__":
    unittest.main()
