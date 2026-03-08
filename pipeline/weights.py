import gdown
from pathlib import Path

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

_AVAILABLE = {
    "FFraw": ("12sLyqBp0VFwdpA-oZLdIOkOTkz_ZnIhV", "FFraw.tar"),
    "FFc23": ("1X0-NYT8KPursLZZdxduRQju6E52hauV0", "FFc23.tar"),
}


def get(name: str) -> str:
    if name not in _AVAILABLE:
        raise ValueError(f"Unknown weights '{name}'. Choose from: {list(_AVAILABLE)}")
    file_id, filename = _AVAILABLE[name]
    dest = _WEIGHTS_DIR / filename
    if not dest.exists():
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {filename} ...")
        gdown.download(id=file_id, output=str(dest), quiet=False)
    return str(dest)


def get_all() -> list[str]:
    return [get(name) for name in _AVAILABLE]


def resolve(path: str | None, name: str = "FFraw") -> str:
    if path:
        return path
    existing = sorted(_WEIGHTS_DIR.glob("*.tar")) if _WEIGHTS_DIR.exists() else []
    return str(existing[0]) if existing else get(name)
