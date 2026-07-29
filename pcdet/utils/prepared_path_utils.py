"""Path normalization for prepared datasets shared across operating systems."""

from pathlib import Path, PureWindowsPath


def resolve_prepared_data_path(root_path, raw_path):
    """Resolve relative prepared-data paths written on Windows or POSIX."""
    root = Path(root_path)
    raw = str(raw_path)
    native = Path(raw)
    if native.is_absolute():
        return native

    windows_path = PureWindowsPath(raw)
    if windows_path.drive:
        return root / windows_path.parent.name / windows_path.name
    return root.joinpath(*windows_path.parts)
