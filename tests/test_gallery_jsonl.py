from pathlib import Path

from cleanup.gallery import jsonl


def test_read_skips_a_line_cut_off_by_a_killed_writer(tmp_path: Path) -> None:
    path: Path = tmp_path / "log.jsonl"
    jsonl.append(path, [{"a": 1}])
    with path.open("a") as f:
        f.write('{"a": 2, "b":')  # killed mid-write: no closing brace, no newline

    jsonl.append(path, [{"a": 3}])

    assert jsonl.read(path) == [{"a": 1}, {"a": 3}]
    assert path.read_text().splitlines()[-1] == '{"a": 3}'  # new row kept on its own line


def test_read_missing_file_is_empty(tmp_path: Path) -> None:
    assert jsonl.read(tmp_path / "missing.jsonl") == []
