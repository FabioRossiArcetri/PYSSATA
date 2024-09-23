import pyssata
pyssata.init(1)
from pyssata.simul import Simul

import cProfile
import pstats
from pstats import SortKey

def main():
    dir = './'
    simul = Simul(dir + 'params_scao_plots.yml')
    simul.run()

if __name__ == '__main__':
    cProfile.run('main()', 'PYSSATA_stats') # , 
    p = pstats.Stats('PYSSATA_stats')
    p.strip_dirs().sort_stats('cumtime').print_stats(50)
