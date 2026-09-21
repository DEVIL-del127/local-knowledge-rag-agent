"""CLI entry point. Dependency composition lives in agent.application."""
from pathlib import Path
import os
import sys


def _bootstrap_source_paths(root: Path) -> None:
    """Make the source-layout NLU package importable without reinstalling the venv.

    ``nlu_v2`` lives under ``kb-agent``.  A wheel/editable install knows that
    mapping from pyproject.toml, but ``python agent_main.py`` must also work from
    a plain Windows or WSL checkout when the existing venv is stale.
    """
    nlu_root = (root / "kb-agent").resolve()
    if not (nlu_root / "nlu_v2" / "__init__.py").is_file():
        raise RuntimeError(f"NLU source package is missing: {nlu_root / 'nlu_v2'}")
    value = str(nlu_root)
    if value not in sys.path:
        sys.path.insert(0, value)


def main() -> None:
    root = Path(os.environ.get("STUDY_PROJECT_ROOT") or Path(__file__).resolve().parent).resolve()
    _bootstrap_source_paths(root)

    from dotenv import load_dotenv
    from agent.app_settings import AppSettings
    from agent.application import build_application, _setup_logging

    load_dotenv(root / ".env")
    _setup_logging()
    build_application(AppSettings.from_env(root)).run()


if __name__ == "__main__":
    main()
