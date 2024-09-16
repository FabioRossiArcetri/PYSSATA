import numpy as np
import os

gpuEnabled = False
cp = None
xp = None

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
