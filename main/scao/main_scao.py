

from pyssata.simul import Simul

def main():
    dir = './'
    simul = Simul(dir + 'params_scao.yml')
    simul.run()


if __name__ == '__main__':
    main()
