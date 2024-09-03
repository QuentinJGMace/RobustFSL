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
    def __init__(self, args, variable_param_dict, **kwargs):
        """

        Args:
            args: base config
            variable_param_dict: dict of parameters to vary {param_name: [param_values]}
        """
        self.args = args
        if "all_methods" in kwargs:
            self.all_methods = kwargs["all_methods"]
        else:
            self.all_methods = False
        self.param_dict = variable_param_dict

        self.init_logger()
        self.init_run()

    def create_new_args(self, new_params):
        if "method" in new_params:
            self.args = parse_args(new_params["method"])
        new_args = copy.deepcopy(self.args)
        for key, value in new_params.items():
            setattr(new_args, key, value)
        return new_args

    def init_run(self):
        self.logger.info("Starting experiment")
        self.logger.info(f"Base config : \n {self.args}")
        self.device = torch.device("cuda" if self.args.cuda else "cpu")

        torch.cuda.set_device(self.args.device)

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
            # try:
            results_run = evaluator.run_full_evaluation(
                backbone=backbone, preprocess=None, return_results=True
            )
            # except Exception as e:
            #     print(e)
            #     self.logger.info(f"An error has occured for the set of parameters : {param_set}")
            #     results_run = None
            global_results.append(results_run)
            params_tested.append(new_params)
            self.logger.info("Finished experiment\n---------------------------------")

        # deletes logger
        self.logger.del_logger()
        return global_results, params_tested


if __name__ == "__main__":
    args = parse_args()
    # fig, ax = plt.subplots(1, 1)

    experiment_dicts = []
    experiment_dicts.append(
        {
            "method": ["tim", "paddle", "mm_rpaddle", "mm_rpaddle_reg", "rpaddle2"],
            "shots": [5],
            "n_outliers_support": [i for i in range(0, 30, 3)],
        }
    )
    experiment_dicts.append(
        {
            "method": ["tim", "paddle", "mm_rpaddle", "mm_rpaddle_reg", "rpaddle2"],
            "shots": [5],
            "n_outliers_query": [i for i in range(0, 35, 5)],
        }
    )
    experiment_dicts.append(
        {
            "method": ["tim", "paddle", "mm_rpaddle", "mm_rpaddle_reg", "rpaddle2"],
            "shots": [10],
            "n_outliers_support": [i for i in range(0, 60, 6)],
        }
    )
    experiment_dicts.append(
        {
            "method": ["tim", "paddle", "mm_rpaddle", "mm_rpaddle_reg", "rpaddle2"],
            "shots": [10],
            "n_outliers_query": [i for i in range(0, 35, 3)],
        }
    )
    experiment_dicts.append(
        {
            "method": ["tim", "paddle", "mm_rpaddle", "mm_rpaddle_reg", "rpaddle2"],
            "shots": [20],
            "n_outliers_support": [i for i in range(0, 100, 10)],
        }
    )
    experiment_dicts.append(
        {
            "method": ["tim", "paddle", "mm_rpaddle", "mm_rpaddle_reg", "rpaddle2"],
            "shots": [20],
            "n_outliers_query": [i for i in range(0, 35, 3)],
        }
    )
    # experiment_dicts.append(
    #     {
    #         "shots": [5],
    #         "beta": np.arange(1.1, 2, 0.1),
    #         "n_outliers_support": [i for i in range(0, 101, 10)],
    #     }
    # )
    # experiment_dicts.append(
    #     {
    #         "shots": [5],
    #         "beta": np.arange(1.1, 2, 0.1),
    #         "n_outliers_query": [i for i in range(0, 76, 10)],
    #     }
    # )
    # experiment_dicts.append(
    #     {
    #         "shots": [20],
    #         "beta": np.arange(1.1, 2, 0.1),
    #         "n_outliers_support": [i for i in range(0, 401, 40)],
    #     }
    # )
    # experiment_dicts.append(
    #     {
    #         "shots": [20],
    #         "beta": np.arange(1.1, 2, 0.1),
    #         "n_outliers_query": [i for i in range(0, 76, 10)],
    #     }
    # )

    for dico in experiment_dicts:
        if "method" in dico:
            experiment = Experiment(args, dico, all_methods=True)
        else:
            experiment = Experiment(args, dico, all_methods=False)
        results, params_tested = experiment.run()
    # fig, ax = plot_evolution_with_param(
    #     param_sets=params_tested_support,
    #     results=results_support,
    #     param_name="n_outliers_support",
    #     fig=fig,
    #     ax=ax,
    # )
    # fig, ax = plot_evolution_with_param(
    #     param_sets=params_tested_query,
    #     results=results_query,
    #     param_name="n_outliers_query",
    #     fig=fig,
    #     ax=ax,
    # )
    # # Saves the figure
    # fig.savefig("plots/evolution_outlier_paddle_5_shot.png")
