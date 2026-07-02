"""lab_utils.compat — small cross-version shims.

NumPy 2.0 removed ``np.trapz`` in favour of ``np.trapezoid``. Use ``trapz``
from here so the code runs on both the newer NumPy on the Ada box and the
older NumPy on the 2080 Ti.
"""

import numpy as np

# np.trapezoid (NumPy >= 2.0) preferred; np.trapz on older NumPy.
trapz = getattr(np, "trapezoid", None) or np.trapz
