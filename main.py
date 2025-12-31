import faulthandler

from Experiment import Experiment

if __name__ == '__main__':
    faulthandler.enable()
    exp1 = Experiment()
    exp1.plan_experiment()
