from pathlib import Path


def explore_dir(root: Path, indent: int = 0) -> None:
    """
    Recursively explore directories until a directory containing '.model' is found.
    That directory is treated as a leaf and not explored further.
    """
    if not root.is_dir():
        return

    prefix = "│   " * indent + "├── "
    print(f"{prefix}{root.name}/")

    # Stop condition: directory contains a '.model' file or folder
    if any(p.name == ".model" for p in root.iterdir()):
        return

    # Recurse into subdirectories (sorted for determinism)
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    for subdir in subdirs:
        explore_dir(subdir, indent + 1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python explore.py <root_directory>")
        sys.exit(1)

    root_path = Path(sys.argv[1]).resolve()
    explore_dir(root_path)
