import os
from pathlib import Path

import pytest

from cleanup.cli import load_env


def test_load_env_exports_dotenv_overriding_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("AWS_BEARER_TOKEN_BEDROCK=from-dotenv\nAWS_REGION=eu-west-1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    load_env()

    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "from-dotenv"
    assert os.environ["AWS_REGION"] == "eu-west-1"
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
