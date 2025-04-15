

import os
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger as Logger
from lightning.pytorch.utilities import rank_zero_only

class LoggerLearning(Logger):
    def __init__(self,name,version):
        super().__init__('lightning_logs',name,version)

    # @property
    # def name(self):
    #     return self._name

    # @property
    # def version(self):
    #     # Return the experiment version, int or str.
    #     return self._vers

    # @rank_zero_only
    # def log_hyperparams(self, params):
    #     # params is an argparse.Namespace
    #     # your code to record hyperparameters goes here
    #     print(params)

    # @rank_zero_only
    # def log_metrics(self, metrics, step):
    #     # metrics is a dictionary of metric names and values
    #     # your code to record metrics goes here
    #     print(metrics,step)


    # @rank_zero_only
    # def save(self):
    #     # Optional. Any code necessary to save logger data goes here
    #     pass

    # @rank_zero_only
    # def finalize(self, status):
    #     # Optional. Any code that needs to be run after training
    #     # finishes goes here
    #     pass
