
import time
import cupy as cp
import numpy as np
from collections import defaultdict


def test_one_size(s1, s2, s3, axes, xp):

    v = xp.ones((s1, s2, s3), dtype=xp.complex64)

    # Warmup
    _ = xp.fft.fft2(v, axes=axes)
    if xp == cp: xp.cuda.runtime.deviceSynchronize()

    t0 = time.time()
    _ = xp.fft.fft2(v, axes=axes)
    if xp == cp: xp.cuda.runtime.deviceSynchronize()
    t1 = time.time()

    return t1 - t0


if __name__ == '__main__':
    s1 = s2 = 1024
    size_list = range(1,10)
    elapsed_list = defaultdict(list)

    for n in size_list:
        elapsed_list[f'CuPy: {s1} x {s2} x n'].append(test_one_size(s1, s2, n, axes=(0, 1), xp=cp))

    for n in size_list:
        elapsed_list[f'CuPy: n x {s1} x {s2}'].append(test_one_size(n, s1, s2, axes=(1, 2), xp=cp))

    for n in size_list:
        elapsed_list[f'NumPy: {s1} x {s2} x n'].append(test_one_size(s1, s2, n, axes=(0, 1), xp=np))

    for n in size_list:
        elapsed_list[f'NumPy: n x {s1} x {s2}'].append(test_one_size(n, s1, s2, axes=(1, 2), xp=np))

    import matplotlib.pyplot as plt
    for k in elapsed_list:
        plt.figure()
        plt.plot(size_list, elapsed_list[k], '.-')
        plt.xlabel(f'FFT size ({k})')
        plt.ylabel('Computation time (s)')
    plt.show()
