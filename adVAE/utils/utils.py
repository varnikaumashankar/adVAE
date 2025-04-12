from pathlib import Path
import argparse

exclude_dirs = {"results", "data", "__pycache__", "adVAE.egg-info", ".gitignore", ".git", ".pytest_cache", ".DS_Store"}

def print_directory_structure(path: Path, indent: str = "", root: Path = None):
    if root is None:
        root = path

    relative = path.relative_to(root)

    # Handle the root directory name properly
    if relative == Path("."):
        print(f"[{path.resolve().name}]")
    elif path.is_dir():
        print(indent + f"[{path.name}]")
    else:
        print(indent + path.name)

    if path.is_dir():
        for item in sorted(path.iterdir()):
            if item.name in exclude_dirs: 
                continue
            print_directory_structure(item, indent + "    ", root)

def main():
    parser = argparse.ArgumentParser(description="Print a directory tree using pathlib.")
    parser.add_argument(
        "--dir", type=str, default=".", help="Directory to print (default: current directory)"
    )
    args = parser.parse_args()
    print_directory_structure(Path(args.dir))

if __name__ == "__main__":
    main()
