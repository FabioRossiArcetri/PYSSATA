# mpiexec -n 2 python script.py args

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import sys

import specula

specula.init(device_idx=int(sys.argv[1]), precision=int(sys.argv[2]), comm=comm, rank=rank)

from specula.simul import Simul

import cProfile
from pstats import Stats

def addRankToIniName(name, r):
    name_no_ext, ext = name.split('.')
    return name_no_ext+str(r)+'.'+ext

def main(*inifiles):
    global comm
    global rank
    param_files = list(inifiles)
    param_files[0] = addRankToIniName(param_files[0], rank)
    print(param_files)
    simul = Simul(*param_files)
    simul.run()

if __name__ == '__main__':
    if sys.argv[3]=='profile':
        with cProfile.Profile() as pr:
            main(*sys.argv[4:])
        stats = Stats(pr).sort_stats("cumtime")
        stats.print_stats(r"\((?!\_).*\)$", 200)
    else:
        main(*sys.argv[3:])
