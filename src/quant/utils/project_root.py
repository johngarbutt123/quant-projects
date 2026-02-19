from pathlib import Path
import os


def set_root():
    cwd = Path.cwd()

    if cwd.name == "notebooks":
        os.chdir(cwd.parent)
    elif (cwd / "quant-projects").exists():
        os.chdir(cwd / "quant-projects")

    return Path.cwd()
