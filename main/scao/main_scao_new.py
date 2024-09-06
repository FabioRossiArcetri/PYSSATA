

from pyssata.simul import Simul


def main():
    dir = '/home/puglisi/git/PYSSATA/main/scao/'
    simul = Simul(dir + 'params_scao_new.py')
    simul.run()


if __name__ == '__main__':
    main()
