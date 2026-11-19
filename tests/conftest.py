"""Gemeinsame Einstellungen: Wurzelordner im Importpfad, ein Thread für BLAS (kleine dichte Lösungen sind sonst 100mal langsamer)."""

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
