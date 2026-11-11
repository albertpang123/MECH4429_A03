"""Make the repository root importable, and defuse a Windows-only abort.

Two things happen here, both because of Windows:

1. **Import path.** pytest loads this before the test modules, so the repository
   root is on ``sys.path`` no matter which directory the command was typed in.

2. **``KMP_DUPLICATE_LIB_OK``.** PyTorch's CPU wheel bundles Intel's OpenMP
   runtime (``libiomp5md.dll``); NumPy and SciPy from conda-forge bring their own
   OpenMP. When both are loaded in one process, Intel's runtime calls ``abort()``
   the next time NumPy reaches LAPACK. The symptom is spectacular and misleading:
   ``Fatal Python error: Aborted`` with the top frame inside
   ``numpy.linalg.inv``, often while ``gym_pybullet_drones`` is parsing a URDF --
   nothing to do with the code being tested, and no Python traceback.

   Nothing here loads torch on its own; pytest's collection phase imports every
   test module first, and ``test_pipeline.py`` imports Stable-Baselines3. That is
   why the abort only shows up when the *whole* suite runs, never for a single
   file. Setting the variable is Intel's documented escape hatch; it silences the
   duplicate-runtime check without changing results.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
