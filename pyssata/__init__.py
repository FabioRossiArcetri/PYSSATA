import numpy as np
import os

cpu_float_dtype_list = [np.float64, np.float32]
cpu_complex_dtype_list = [np.complex128, np.complex64]

gpuEnabled = False
cp = None
xp = None
global_precision = None
float_dtype_list = None
complex_dtype_list = None
gpu_float_dtype_list = cpu_float_dtype_list
gpu_complex_dtype_list = cpu_complex_dtype_list
float_dtype = None
complex_dtype = None
default_target_device_idx = None
default_target_device = None

# precision = 0 -> double precision
# precision = 1 -> single precision

# target_device = -1 -> CPU
# target_device = i>-1 -> GPUi

# you might have a GPU working and cupy installed
# and still want to use the CPU (idx==-1) as default_target_device
# in this case you might still want to allocate some objects on
# a GPU device (idx>=0).
# This can be checked later looking at the  value of gpuEnabled.

def init(device_idx=-1, precision=0):
    global xp
    global cp
    global gpuEnabled
    global global_precision
    global float_dtype_list
    global complex_dtype_list
    global gpu_float_dtype_list
    global gpu_complex_dtype_list
    global float_dtype
    global complex_dtype
    global default_target_device_idx
    global default_target_device
    
    default_target_device_idx = device_idx
    systemDisable = os.environ.get('PYSSATA_DISABLE_GPU', 'FALSE')
    if systemDisable=='FALSE':
        try:
            import cupy as cp
            print("Cupy import successfull. Installed version is:", cp.__version__)
            gpuEnabled = True
        except:
            print("Cupy import failed. PYSSATA will fall back to CPU use.")
            cp = np
            xp = np
            default_target_device_idx=-1
    else:
        print("env variable PYSSATA_DISABLE_GPU prevents using the GPU.")
        cp = np
        xp = np
        default_target_device_idx=-1


    if default_target_device_idx>=0:
        xp = cp
        gpu_float_dtype_list = [cp.float64, cp.float32]
        gpu_complex_dtype_list = [cp.complex128, cp.complex64]
        default_target_device = cp.cuda.Device(default_target_device_idx)
        default_target_device.use()
        print('Default device is GPU number ', default_target_device_idx)
        # print('Using device: ', cp.cuda.runtime.getDeviceProperties(default_target_device)['name'])
        # attributes = default_target_device.attributes
        # properties = cp.cuda.runtime.getDeviceProperties(default_target_device)
        # print('Number of multiprocessors:', attributes['MultiProcessorCount'])
        # print('Global memory size (GB):', properties['totalGlobalMem'] / (1024**3))
    else:
        print('Default device is CPU')
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
        # which one is better, xp.asnumpy(v) or v.get() ? almost the same but asnumpy is more general
        return xp.asnumpy(v)


