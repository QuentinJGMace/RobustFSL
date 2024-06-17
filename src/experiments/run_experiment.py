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


class Experiment:
    def __init__(self, args, variable_param_dict):
        """

        Args:
            args: base config
            variable_param_dict: dict of parameters to vary {param_name: [param_values]}
        """
        self.args = args
        self.param_dict = variable_param_dict

        self.init_logger()
        self.init_run()

    def create_new_args(self, new_params):
        new_args = copy.deepcopy(self.args)
        for key, value in new_params.items():
            setattr(new_args, key, value)
        return new_args

    def init_run(self):
        self.logger.info("Starting experiment")
        self.device = torch.device("cuda" if self.args.cuda else "cpu")

        if self.args.seed != -1:
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            cudnn.deterministic = True
        torch.cuda.set_device(self.args.device)

    def init_logger(self):
        self.log_file = get_log_file(
            log_path=self.args.log_path,
            dataset=self.args.dataset,
            method=self.args.name_method,
        )
        self.logger = Logger(__name__, self.log_file)
        self.logger.info("Logger initialized")

    def load_backbone(self):
        backbone = get_backbone(self.args).to(self.device)
        checkpoint_path = self.args.ckpt_path
        load_checkpoint(backbone, checkpoint_path, self.device, type="best")

        return backbone

    def run(self):

        backbone = self.load_backbone()

        global_results = []
        params_tested = []

        # Chooses a param set to perform grid search
        for param_set in itertools.product(*self.param_dict.values()):
            new_params = dict(zip(self.param_dict.keys(), param_set))
            new_args = self.create_new_args(new_params)

            # run experiment
            if new_args.shots > 0:
                evaluator = Evaluator_few_shot(
                    device=self.device,
                    args=new_args,
                    log_file=self.log_file,
                    logger=self.logger,
                    disable_tqdm=True,
                )
            else:
                raise ValueError("Invalid number of shots")

            self.logger.info(f"Running experiment with params: {new_params}")
            # IMPORTANT : Feature extraction should have already been performed previously !
            results_run = evaluator.run_full_evaluation(
                backbone=backbone, preprocess=None, return_results=True
            )
            global_results.append(results_run)
            params_tested.append(new_params)
            self.logger.info("Finished experiment\n---------------------------------")

        # deletes logger
        self.logger.del_logger()
        return global_results, params_tested


if __name__ == "__main__":
    args = parse_args()
    fig, ax = plt.subplots(1, 1)
    variable_param_dict_support = {
        "shots": [5],
        "n_outliers_support": [i for i in range(0, 101, 5)],
    }
    experiment_support = Experiment(args, variable_param_dict_support)
    results_support, params_tested_support = experiment_support.run()
    fig, ax = plot_evolution_with_param(
        param_sets=params_tested_support,
        results=results_support,
        param_name="n_outliers_support",
        fig=fig,
        ax=ax,
    )

    variable_param_dict_query = {
        "shots": [5],
        # "n_outliers_support": [0],  # We keep the same number of outliers in the support set
        "n_outliers_query": [i for i in range(0, 76, 5)],
    }
    experiment_query = Experiment(args, variable_param_dict_query)
    results_query, params_tested_query = experiment_query.run()
    fig, ax = plot_evolution_with_param(
        param_sets=params_tested_query,
        results=results_query,
        param_name="n_outliers_query",
        fig=fig,
        ax=ax,
    )
    # Saves the figure
    fig.savefig("evolution_outlier_rpaddle_rand.png")
