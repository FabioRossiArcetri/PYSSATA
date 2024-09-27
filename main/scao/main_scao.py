import sys

import pyssata

pyssata.init(device_idx=int(sys.argv[1]), precision=0)

from pyssata.simul import Simul

import cProfile
from pstats import Stats

def main(inifile):
    simul = Simul(inifile)
    simul.run()

if __name__ == '__main__':    
    with cProfile.Profile() as pr:
        main(sys.argv[2])    
    stats = Stats(pr).sort_stats("cumtime")
    stats.print_stats(r"\((?!\_).*\)$", 20)
    