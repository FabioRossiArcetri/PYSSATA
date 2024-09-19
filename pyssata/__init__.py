import numpy as np
import os

gpuEnabled = False
cp = None
xp = None
standard_dtype = None
complex_dtype = None

def init(precision):
    global standard_dtype
    global complex_dtype
    global xp
    global cp
    global gpuEnabled
    systemDisable = os.environ.get('PYSSATA_DISABLE_GPU', 'FALSE')
    if systemDisable=='FALSE':
        try:
            import cupy as cp
            print("Cupy import successfull. Installed version is:", cp.__version__)
            gpuEnabled = True
            xp = cp
        except:
            print("Cupy import failed. PYSSATA will fall back to CPU use.")
            cp = np
            xp = np
    else:
        print("env variable PYSSATA_DISABLE_GPU prevents using the GPU.")
        cp = np
        xp = np

    if precision==32:
        standard_dtype = xp.float32
        complex_dtype = xp.complex64
    else:
        standard_dtype = xp.float64
        complex_dtype = xp.complex128

def cpuArray(v):
    if isinstance(v, (np.ndarray, np.float64, np.int64)):
        return v
    else:
        return v.get()

