""" 

module to ensure the existence of paths

"""


from pathlib import Path

BASE_PATH = (Path(__file__).parent.parent.parent / "files").expanduser().resolve()

# === GENERAL ===

def get_base_path(ensure: bool = True) -> Path:
    """Get the base path for storing files."""
    if ensure:
        BASE_PATH.mkdir(parents=True, exist_ok=True)
    return BASE_PATH


# === STEADY ===

def ensure_steady_scanning_paths(base_path: Path = BASE_PATH) -> None:
    """Ensure the existence of paths for steady scanning results."""
    steady_path = base_path / "steady_scanning"
    steady_path.mkdir(parents=True, exist_ok=True)

def get_steady_scanning_path(level: int = 0, base_path: Path = BASE_PATH) -> Path:
    """Get the path for steady scanning results for a specific level."""
    return base_path / "steady_scanning" / f"level{level}.txt"


# === TEMPORAL ===

def ensure_temporal_scanning_paths(base_path: Path = BASE_PATH) -> None:
    """Ensure the existence of paths for temporal scanning results."""
    temporal_path = base_path / "temporal_scanning"
    temporal_path.mkdir(parents=True, exist_ok=True)
    for case in ["A", "B", "C", "D", "E"]:
        case_path = temporal_path / f"case_{case}"
        case_path.mkdir(parents=True, exist_ok=True)
        for quantity in ["Ca", "gs", "K"]:
            quantity_path = case_path / quantity
            quantity_path.mkdir(parents=True, exist_ok=True)

def get_case_path(case: str, base_path: Path = BASE_PATH) -> Path:
    """Get the path for a specific case."""
    return base_path / "temporal_scanning" / f"case_{case}"


def get_temporal_scanning_path(case: str, quantity: str, base_path: Path = BASE_PATH) -> Path:
    """Get the path for temporal scanning results for a specific case and quantity."""
    return get_case_path(case, base_path=base_path) / quantity

