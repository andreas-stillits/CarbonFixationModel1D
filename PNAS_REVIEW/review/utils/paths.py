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

