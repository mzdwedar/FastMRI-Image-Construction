"""Where the Hydra configs live.

In a git checkout / editable install they are at the repo root (two levels above this package). In a wheel
(e.g. `pip install git+https://...` on Colab) the build copies them into the package; see pyproject.toml.
The value is relative to this package directory, which is what `hydra.main` resolves against for modules here.
"""

from pathlib import Path

CONFIG_PATH = "configs" if (Path(__file__).parent / "configs").is_dir() else "../../configs"
