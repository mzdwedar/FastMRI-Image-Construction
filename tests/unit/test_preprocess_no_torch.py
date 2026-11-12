import subprocess
import sys

CODE = """
import sys
sys.modules["torch"] = None  # any `import torch` now raises ImportError
import fastmri_knee.preprocess as p
import numpy as np
assert p.kspace_to_image(np.ones((640, 368), dtype=np.complex64)).shape == (1, 320, 320)
"""


def test_preprocess_imports_without_torch():
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
