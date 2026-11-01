"""Higiene do repositório público: nada privado ou secreto entra no Git.

Varre os arquivos versionados e os novos que ainda não são ignorados (o que um `git add .`
levaria). Os relatórios dizem arquivo e linha, nunca o valor encontrado.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from sentinel.security import find_secrets

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1_000_000
PRIVATE_NAMES = {".env", "CLAUDE.md", "data-sentinel-CLAUDE.md", "ground_truth.json"}
PRIVATE_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pem", ".key"}
PRIVATE_DIRS = {"data", "logs", ".claude", ".venv"}
SECRET_KEY_NAMES = re.compile(r"(TOKEN|DSN|PASSWORD|SECRET|API_KEY|PRIVATE_KEY)", re.I)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )


@pytest.fixture(scope="module")
def shareable_files() -> list[Path]:
    """Arquivos que iriam para o repositório: versionados + novos não ignorados."""
    if shutil.which("git") is None or git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("sem git ou fora de um repositório")
    listing = git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    paths = [ROOT / name for name in listing.stdout.split("\0") if name]
    return [p for p in paths if p.is_file()]


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_no_private_file_would_be_shared(shareable_files):
    offenders = [
        relative(p)
        for p in shareable_files
        if p.name in PRIVATE_NAMES
        or p.suffix in PRIVATE_SUFFIXES
        or (p.name.startswith(".env") and p.name != ".env.example")
        or PRIVATE_DIRS & set(p.relative_to(ROOT).parts[:1])
    ]
    assert offenders == []


def test_no_secret_or_personal_path_in_shared_files(shareable_files):
    problems = []
    for path in shareable_files:
        if path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binário
        problems += [f"{relative(path)}:{line} ({kind})" for kind, line in find_secrets(text)]
    assert problems == []


def test_env_example_has_no_values_for_secret_keys():
    filled = []
    for number, line in enumerate((ROOT / ".env.example").read_text("utf-8").splitlines(), 1):
        key, _, value = line.partition("=")
        if SECRET_KEY_NAMES.search(key) and value.split("#")[0].strip():
            filled.append(f".env.example:{number} ({key.strip()})")
    assert filled == []


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "CLAUDE.md",
        "data-sentinel-CLAUDE.md",
        "data/sentinel.db",
        "logs/audit.jsonl",
        "x.sqlite",
        "chave.pem",
        ".claude/settings.local.json",
    ],
)
def test_gitignore_blocks_private_paths(path):
    if shutil.which("git") is None or git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("sem git ou fora de um repositório")
    assert git("check-ignore", "-q", path).returncode == 0, f"{path} não está no .gitignore"


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "sentinel/data/generator.py",
        "sentinel/data/db.py",
        "scenarios/scenarios.jsonl",
    ],
)
def test_gitignore_does_not_hide_project_files(path):
    if shutil.which("git") is None or git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("sem git ou fora de um repositório")
    assert git("check-ignore", "-q", path).returncode == 1, f"{path} está sendo ignorado"
