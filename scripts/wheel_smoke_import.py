"""Import an installed wheel target without starting application services."""
from pathlib import Path
import sys


def main(target: str) -> None:
    root = Path(target).resolve()
    if not root.is_dir():
        raise SystemExit("wheel target missing")
    sys.path.insert(0, str(root))
    import agent_main
    import agent.application
    import core.retrieval_gateway
    import ingestion.pipeline.catalog_migration
    import memory.memory_manager
    import nlu_v2
    for module in (agent_main, agent.application, core.retrieval_gateway,
                   ingestion.pipeline.catalog_migration, memory.memory_manager, nlu_v2):
        path = Path(module.__file__).resolve()
        if root not in path.parents:
            raise SystemExit(f"source tree shadowed installed module: {module.__name__}")
    print("wheel imports ok")


if __name__ == "__main__":
    main(sys.argv[1])
