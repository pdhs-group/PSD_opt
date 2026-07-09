# PSD_opt WMCPBE Paper Branch

This branch is dedicated to the paper:

**"A Weighted Monte Carlo Solver for Multidimensional Population Balance Equations with Dynamic Grid-Based Reconstruction"**

It preserves the source code for the paper-related models and case scripts. The main WMCPBE solver source code is located in:

```text
mcpbe/src/wmcpbe
```

The case scripts are located in:

```text
scripts
```

**Note: Please manually install openpyxl in your current Python environment.**
```powershell
python -m pip install openpyxl
```

## Installation Option 1: Editable Install from the Repository

Use this option if you want to copy or clone the whole repository locally and link the source code directly into your current Python environment.

Prerequisite: install Poetry into the same Python environment before running the build script.

```powershell
python -m pip install poetry
git clone https://github.com/pdhs-group/PSD_opt.git
cd PSD_opt
python build_all.py --editable
```

If you already have a local copy of this branch, run the editable install from the repository root:

```powershell
python -m pip install poetry
cd C:\path\to\PSD_opt
python build_all.py --editable
```

The editable install links the package sources into the active Python environment, so later source-code changes are picked up without reinstalling the wheels.

## Installation Option 2: Install Prebuilt Wheels

Use this option if you only want to install the released packages.

Download the following four wheel files from the WMCPBE release page:

<https://github.com/pdhs-group/PSD_opt/releases/tag/WMCPBE>

Install them in this order:

1. `pbe_core`
2. `dpbe`
3. `mcpbe`
4. `qmom`

After downloading the four wheels into one folder, open PowerShell in that folder and run:

```powershell
python -m pip install (Get-ChildItem . -Filter "pbe_core-*.whl" | Select-Object -First 1).FullName
python -m pip install (Get-ChildItem . -Filter "dpbe-*.whl" | Select-Object -First 1).FullName
python -m pip install (Get-ChildItem . -Filter "mcpbe-*.whl" | Select-Object -First 1).FullName
python -m pip install (Get-ChildItem . -Filter "qmom-*.whl" | Select-Object -First 1).FullName
```

For Bash terminals, run the same installation order with shell wildcards:

```bash
python -m pip install ./pbe_core-*.whl
python -m pip install ./dpbe-*.whl
python -m pip install ./mcpbe-*.whl
python -m pip install ./qmom-*.whl
```
