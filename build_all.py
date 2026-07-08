from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PACKAGE_INSTALL_ORDER = ("pbe-core", "dpbe", "mcpbe", "qmom")


def package_sort_key(package_dir: Path) -> tuple[int, str]:
    try:
        order_index = PACKAGE_INSTALL_ORDER.index(package_dir.name)
    except ValueError:
        order_index = len(PACKAGE_INSTALL_ORDER)
    return (order_index, package_dir.name.lower())


def find_package_dirs(root: Path) -> list[Path]:
    package_dirs: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith('.'):
            continue
        if child.name in {"dist-wheels", "dist", "__pycache__"}:
            continue
        if (child / "pyproject.toml").exists():
            package_dirs.append(child)
    return sorted(package_dirs, key=package_sort_key)


def run_poetry_build(package_dir: Path) -> None:
    cmd = [sys.executable, "-m", "poetry", "build", "-f", "wheel"]
    result = subprocess.run(cmd, cwd=package_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Build failed for {package_dir.name}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def check_poetry_module() -> bool:
    cmd = [sys.executable, "-m", "poetry", "--version"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        version_text = result.stdout.strip() or result.stderr.strip()
        print(f"[INFO] Using Poetry from current Python: {sys.executable}")
        if version_text:
            print(f"[INFO] {version_text}")
        return True

    print(f"[ERROR] Poetry is not available from current Python: {sys.executable}")
    print("Install it into this environment with:")
    print(f'  "{sys.executable}" -m pip install poetry')
    if result.stderr.strip():
        print("stderr:")
        print(result.stderr.strip())
    return False


def run_editable_install(package_dir: Path) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-e", str(package_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Editable install failed for {package_dir.name}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def collect_wheels(package_dir: Path, out_dir: Path) -> list[Path]:
    dist_dir = package_dir / "dist"
    wheels = sorted(dist_dir.glob("*.whl"))
    moved: list[Path] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        dst = out_dir / wheel.name
        shutil.move(str(wheel), str(dst))
        moved.append(dst)
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build wheels for all first-level package folders with pyproject.toml"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root directory (default: script directory)",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete dist-wheels before collecting new wheels",
    )
    parser.add_argument(
        "--editable",
        action="store_true",
        help="Install all discovered packages in editable mode into current environment",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = root / "dist-wheels"

    if not args.editable and not check_poetry_module():
        return 1

    if args.clean_output and out_dir.exists() and not args.editable:
        shutil.rmtree(out_dir)

    package_dirs = find_package_dirs(root)
    if not package_dirs:
        print("[INFO] No package folders with pyproject.toml found.")
        return 0

    print(f"[INFO] Found {len(package_dirs)} package folder(s).")

    all_wheels: list[Path] = []
    editable_installed: list[str] = []
    failed: list[tuple[str, str]] = []

    for pkg in package_dirs:
        print(f"\n[BUILD] {pkg.name}")
        try:
            if args.editable:
                run_editable_install(pkg)
                editable_installed.append(pkg.name)
                print(f"  [OK] editable installed: {pkg.name}")
            else:
                run_poetry_build(pkg)
                wheels = collect_wheels(pkg, out_dir)
                if not wheels:
                    print("  [WARN] Build finished but no wheel found in dist/.")
                else:
                    for w in wheels:
                        print(f"  [OK] {w.name}")
                    all_wheels.extend(wheels)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {exc}")
            failed.append((pkg.name, str(exc)))

    print("\n================ Summary ================")
    if args.editable:
        print(f"Editable installations: {len(editable_installed)}")
    else:
        print(f"Output directory: {out_dir}")
        print(f"Total wheels moved: {len(all_wheels)}")
    if failed:
        print(f"Failed packages: {len(failed)}")
        for name, msg in failed:
            print(f"  - {name}: {msg.splitlines()[0]}")
        return 2

    print("All package wheels built successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
