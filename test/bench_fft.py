
import time
import cupy as cp
from collections import defaultdict


def test_one_size(s1, s2, s3, axes):

    v = cp.ones((s1, s2, s3), dtype=cp.complex64)

    # Warmup
    _ = cp.fft.fft2(v, axes=axes)
    cp.cuda.runtime.deviceSynchronize()

    t0 = time.time()
    _ = cp.fft.fft2(v, axes=axes)
    cp.cuda.runtime.deviceSynchronize()
    t1 = time.time()

    return t1 - t0


s1 = s2 = 1024
size_list = range(1,60)
elapsed_list = defaultdict(list)

for n in size_list:
    elapsed_list[f'{s1} x {s2} x n'].append(test_one_size(s1, s2, n, axes=(0, 1)))

for n in size_list:
    elapsed_list[f'n x {s1} x {s2}'].append(test_one_size(n, s1, s2, axes=(1, 2)))

import matplotlib.pyplot as plt
for k in elapsed_list:
    plt.figure()
    plt.plot(size_list, elapsed_list[k], '.-')
    plt.xlabel(f'FFT size ({k})')
    plt.ylabel('Computation time (s)')
plt.show()
