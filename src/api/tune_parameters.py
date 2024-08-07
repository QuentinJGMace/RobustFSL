import torch

from src.api.utils import parse_args
from src.experiments.run_experiment import Experiment


def tune_parameters(args, params_to_tune):
    """
    Tune the parameters of the experiment

    Args:
        args : Namespace, arguments of the experiment
        params_to_tune : dict, parameters to tune {param_name: [param_values]}

    Returns:
        best_params : dict, best parameters
        best_acc : float, best accuracy
        results : dict, results of the experiment
    """

    experiment = Experiment(args, params_to_tune)
    results, params_tested = experiment.run()

    best_acc = -1
    best_params = None
    for param, result in zip(params_tested, results):
        if result is None:
            continue
        if result["mean_accuracies"] > best_acc:
            best_acc = result["mean_accuracies"]
            best_params = param

    return best_params, best_acc, results


if __name__ == "__main__":
    args = parse_args()
    params_to_tune = {
        "iter": [2, 5, 10],
        "lambd": [75],
        "beta": [1.5],
        "kappa": [0, 10, 50],
    }
    best_params, best_acc, results = tune_parameters(args, params_to_tune)
    print(f"Best parameters : {best_params}")
    print(f"Best accuracy : {best_acc}")
