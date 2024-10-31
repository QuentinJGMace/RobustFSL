import copy
import itertools
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn

from src.backbones.utils import get_backbone, load_checkpoint
from src.logger.logger import Logger, get_log_file
from src.api.eval_few_shot import Evaluator_few_shot
from src.api.utils import parse_args
from src.experiments.plot_utils import plot_evolution_with_param


class Experiment2:
    def __init__(self, path_main_config, path_variable_config, **kwargs):
        """
        Args:
            path_main_config: path to the main config file
            path_variable_config: path to the variable config file
        """
        self.path_main_config = path_main_config
        self.path_variable_config = path_variable_config
        self.load_config_files(path_main_config, path_variable_config)
        self.init_logger()
        self.init_run()

    def init_logger(self):
        if not self.all_methods:
            self.log_file = get_log_file(
                log_path=self.args.log_path,
                dataset=self.args.dataset,
                method=self.args.name_method,
            )
        else:
            self.log_file = get_log_file(
                log_path=self.args.log_path,
                dataset=self.args.dataset,
                method="all",
            )
        self.logger = Logger(__name__, self.log_file)
        self.logger.info("Logger initialized")

    def init_run(self):
        self.logger.info("Starting experiment")
        self.logger.info(f"Base config : \n {self.args}")
        self.device = torch.device("cuda" if self.args.cuda else "cpu")

        torch.cuda.set_device(self.args.device)

    def load_config_files(self, path_main_config, path_variable_config):
        self.args = parse_args(path_main_config)
        self.variable_param_dict = parse_args(path_variable_config)
