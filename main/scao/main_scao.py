import sys
import os

#os.chdir('/home/frossi/dev/SPECULA/main/scao')
#sys.argv=['', '0', '1', 'params_scao_sh1.yml']

import specula

specula.init(device_idx=int(sys.argv[1]), precision=int(sys.argv[2]))

from specula.simul import Simul

import cProfile
from pstats import Stats

def main(*inifiles):
    simul = Simul(*inifiles)
    simul.run()

if __name__ == '__main__':
    if sys.argv[3]=='profile':
        with cProfile.Profile() as pr:
            main(*sys.argv[4:])
        stats = Stats(pr).sort_stats("cumtime")
        stats.print_stats(r"\((?!\_).*\)$", 200)
    else:
        main(*sys.argv[3:])
