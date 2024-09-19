import numpy as np
import os

gpuEnabled = False
cp = None
xp = None
global_precision = None
float_dtype_list = None
complex_dtype_list = None
float_dtype = None
complex_dtype = None

# precision = 0 -> double precision
# precision = 1 -> single precision

def init(precision=0):
    global xp
    global cp
    global gpuEnabled
    global global_precision
    global float_dtype_list
    global complex_dtype_list
    global float_dtype
    global complex_dtype
    
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

    float_dtype_list = [xp.float64, xp.float32]
    complex_dtype_list = [xp.complex128, xp.complex64]
    global_precision = precision
    float_dtype = float_dtype_list[global_precision]
    complex_dtype = complex_dtype_list[global_precision]


def cpuArray(v):
    if isinstance(v, (np.ndarray, np.float64, np.int64)):
        return v
    else:
        return v.get()

