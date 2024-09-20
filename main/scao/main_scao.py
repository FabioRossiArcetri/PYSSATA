import pyssata
pyssata.init(device_idx=1, precision=0)

from pyssata.simul import Simul

import cProfile
import pstats
from pstats import SortKey

def main():
    dir = './'
    simul = Simul(dir + 'params_scao.yml')
    simul.run()

if __name__ == '__main__':
    #cProfile.run('main()', 'PYSSATA_stats') # , 
    #p = pstats.Stats('PYSSATA_stats')
    #p.strip_dirs().sort_stats(-1).print_stats(10)
    main()
