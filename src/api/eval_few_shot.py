import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

# import utils and datasets
from src.logger import Logger
from src.api.utils import (
    wrap_tqdm,
    save_pickle,
    load_pickle,
)
from src.api.metric_utils import compute_confidence_interval
from src.api.task_generator_few_shot import Task_Generator_Few_shot
from src.api.sampler_few_shot import (
    CategoriesSampler_few_shot,
    SamplerSupportAndQuery,
)
from src.api.extract_features import extract_features
from src.dataset import DATASET_LIST
from src.dataset import build_data_loader
from src.methods import get_method_builder


class Evaluator_few_shot:
    def __init__(self, device, args, log_file, logger, disable_tqdm=False):
        self.device = device
        self.args = args
        self.save_mult_outlier_distrib = args.save_mult_outlier
        if self.save_mult_outlier_distrib:
            self.true_mults = {"support": [], "query": []}
            self.pred_mults = {"support": [], "query": []}
        self.log_file = log_file
        self.logger = logger
        # self.logger = Logger(__name__, log_file)
        self.disable_tqdm = disable_tqdm

    def initialize_data_loaders(self, dataset, preprocess):
        """
        Initialisises the data loaders

        Args:
            dataset: The dataset to be used.
            preprocess: Preprocessing function for data.

        Return:
            data_loaders: Data loaders for the dataset.
                        (dict : {"train": train_loader, "val": val_loader, "test": test_loader})
        """
        # if batch size is an argument, use it, otherwise use the default batch size to 1024
        batch_size = self.args.batch_size if self.args.batch_size else 1024

        data_loaders = {
            "train": build_data_loader(
                data_source=dataset.train_x,
                batch_size=batch_size,
                is_train=False,
                shuffle=False,
                tfm=preprocess,
            ),
            "val": build_data_loader(
                data_source=dataset.val,
                batch_size=batch_size,
                is_train=False,
                shuffle=False,
                tfm=preprocess,
            ),
            "test": build_data_loader(
                data_source=dataset.test,
                batch_size=batch_size,
                is_train=False,
                shuffle=False,
                tfm=preprocess,
            ),
        }

        return data_loaders

    def extract_and_load_features(self, backbone, data_loaders):
        """
        Extracts and loads the features for evaluation

        Args :
            backbone : The model to be evaluated.
            dataset : The dataset to be used.
            data_loaders : Data loaders for the dataset.
                        (dict : {"train": train_loader, "val": val_loader, "test": test_loader})

        Returns:
            extracted_features_dic_support : Extracted features for support set.
            extracted_features_dic_query : Extracted features for query set.
        """

        if not os.path.exists(
            f"data/{self.args.dataset}/saved_features/train_features_{self.args.backbone}.pkl"
        ):
            extract_features(self.args, backbone, data_loaders["train"], "train")
        if not os.path.exists(
            f"data/{self.args.dataset}/saved_features/val_features_{self.args.backbone}.pkl"
        ):
            extract_features(self.args, backbone, data_loaders["val"], "val")
        if not os.path.exists(
            f"data/{self.args.dataset}/saved_features/test_features_{self.args.backbone}.pkl"
        ):
            extract_features(self.args, backbone, data_loaders["test"], "test")

        filepath_support = os.path.join(
            f"data/{self.args.dataset}/saved_features/{self.args.used_set_support}_features_{self.args.backbone}.pkl"
        )
        filepath_query = os.path.join(
            f"data/{self.args.dataset}/saved_features/{self.args.used_set_query}_features_{self.args.backbone}.pkl"
        )

        extracted_features_dic_support = load_pickle(filepath_support)
        extracted_features_dic_query = load_pickle(filepath_query)

        return extracted_features_dic_support, extracted_features_dic_query

    def run_full_evaluation(self, backbone, preprocess, return_results=False):
        """
        Run the full evaluation process over all tasks.
        :param backbone: The model to be evaluated.
        :param preprocess: Preprocessing function for data.
        :return: Mean accuracies of the evaluation.
        """

        backbone.eval()

        # Init dataset and data loaders
        dataset = DATASET_LIST[self.args.dataset](self.args.dataset_path)
        self.args.classnames = dataset.classnames
        data_loaders = self.initialize_data_loaders(dataset, preprocess)

        # Extract and load features (load them if already precomputed)
        (
            extracted_features_dic_support,
            extracted_features_dic_query,
        ) = self.extract_and_load_features(backbone, data_loaders)
        features_support = extracted_features_dic_support["features"]
        labels_support = extracted_features_dic_support["labels"]
        features_query = extracted_features_dic_query["features"]
        labels_query = extracted_features_dic_query["labels"]

        # Run evaluation for each task and collect results
        mean_accuracies, mean_times, mean_criterions = self.evaluate_tasks(
            backbone, features_support, labels_support, features_query, labels_query
        )

        self.report_results(mean_accuracies, mean_times)

        if return_results:
            return {
                "mean_accuracies": mean_accuracies,
                "mean_times": mean_times,
                "mean_criterions": mean_criterions,
            }

    def set_method_opt_param(self):
        """
        Set the optimal parameter for the method if the test set is used
        """
        raise NotImplementedError("Optimal parameter setting not implemented yet")

    def save_mult(self, mult, is_support, is_true):
        """Save the mults distrib used for outlier generation

        Args :
            mult : the mults used for outlier generation shape (n_outliers, 1)
            is_support : if the mults are used for support or query set
            is_true: if the mults are the true mults or the predicted ones
        """
        path = "results_few_shot/{}/{}".format(
            self.args.used_test_set, self.args.dataset
        )

        set_name = "support" if is_support else "query"
        pred_name = "predicted" if not is_true else "true"
        filename = os.path.join(path, f"{pred_name}_mult_{set_name}.png")

        n_outliers = mult.shape[0]

        plt.figure()
        plt.hist(mult[:, 0].cpu().numpy(), bins=50)
        plt.title(
            f"{pred_name} mults distribution for {n_outliers} outliers in {set_name} set"
        )
        plt.xlabel("Mult")
        plt.ylabel("Count")
        plt.savefig(filename)

    # TODO : improve, add feature from completely different image, switch labels
    def generate_outliers(
        self, all_features, all_labels, indices, n_outliers=0, is_support=False
    ):
        """
        Randomly generates outliers for the given features and labels.

        Args:
            all_features : Extracted features for the dataset.
            all_labels : Labels for the dataset.
            indices : Indices of the data points that will be taken into the set

        Returns:
            new_features : Features with outliers.
            new_labels : Labels with outliers.
            indices_outliers : Indices of the outliers.
        """
        # if no outliers to be added, return the original features and labels
        if (n_outliers) == 0:
            return all_features[indices].clone(), all_labels[indices].clone(), []

        # Else chooses random indices and apply transformations

        # init new tensors
        new_features = torch.zeros_like(all_features[indices])
        new_labels = all_labels[indices].clone()

        # Select random indices
        perm = torch.randperm(len(indices))
        indices_outliers = indices[perm][:n_outliers]

        # Mask for torch magic to optimize runtime
        mask_outliers = torch.zeros(len(indices), dtype=torch.bool)
        mask_outliers[perm[:n_outliers]] = True

        # Replace the outliers with random vectors
        # new_features[mask_outliers] = torch.randn_like(
        #     all_features[indices[mask_outliers]]
        # )
        # new_features[mask_outliers] = torch.zeros_like(
        #     all_features[indices[mask_outliers]]
        # )
        mult = 4 * torch.rand((n_outliers, 1), device=all_features.device) + 1
        new_features[mask_outliers] = mult * all_features[indices[mask_outliers]]
        new_features[~mask_outliers] = all_features[indices[~mask_outliers]]

        if self.save_mult_outlier_distrib and is_support:
            self.true_mults["support"].append(mult)
        elif self.save_mult_outlier_distrib and not is_support:
            self.true_mults["query"].append(mult)

        return new_features, new_labels, indices_outliers

    def evaluate_tasks(
        self, backbone, features_support, labels_support, features_query, labels_query
    ):
        """
        Evaluates the method

        Args :
            backbone : The model to be evaluated.
            features_support : Extracted features for support set.
            labels_support : Labels for support set.
            features_query : Extracted features for query set.
            labels_query : Labels for query set.

        Returns :
            mean_accuracies : Mean accuracies of the evaluation.
            mean_times : Mean time taken for evaluation.
        """

        self.logger.info(
            f"Running evaluation with method {self.args.name_method} on {self.args.dataset} dataset ({self.args.used_set_query} set)"
        )

        results_task = []
        results_task_time = []
        results_criterion = defaultdict(lambda: [])

        # Evaluation over each task
        for i in wrap_tqdm(
            range(int(self.args.number_tasks / self.args.batch_size)),
            disable_tqdm=self.disable_tqdm,
        ):

            # Samplers
            sampler = CategoriesSampler_few_shot(
                self.args.batch_size,
                self.args.k_eff,
                self.args.n_class,
                self.args.shots,
                self.args.n_query,
                force_query_size=True,
            )
            sampler.create_list_classes(labels_support, labels_query)

            sampler_support_query = SamplerSupportAndQuery(sampler)

            test_loader_query, test_loader_support = [], []
            list_indices_support, list_indices_query = [], []
            list_indices_outlier_support, list_indices_outlier_query = [], []

            for indices_support, indices_query in sampler_support_query:
                (
                    new_features_s,
                    new_labels_s,
                    indices_outliers_s,
                ) = self.generate_outliers(
                    features_support,
                    labels_support,
                    indices_support,
                    self.args.n_outliers_support,
                    is_support=True,
                )
                test_loader_support.append((new_features_s, new_labels_s))
                (
                    new_features_q,
                    new_labels_q,
                    indices_outliers_q,
                ) = self.generate_outliers(
                    features_query,
                    labels_query,
                    indices_query,
                    self.args.n_outliers_query,
                    is_support=False,
                )
                test_loader_query.append((new_features_q, new_labels_q))
                list_indices_query.append(indices_query)
                list_indices_support.append(indices_support)
                list_indices_outlier_query.append(indices_outliers_q)
                list_indices_outlier_support.append(indices_outliers_s)

            # Prepare the tasks
            task_generator = Task_Generator_Few_shot(
                k_eff=self.args.k_eff,
                shot=self.args.shots,
                n_query=self.args.n_query,
                n_class=self.args.n_class,
                loader_support=test_loader_support,
                loader_query=test_loader_query,
                backbone=backbone,
                args=self.args,
            )

            tasks = task_generator.generate_tasks()

            # Load the method
            method = get_method_builder(
                backbone=backbone,
                device=self.device,
                args=self.args,
                log_file=self.log_file,
            )
            # set the optimal parameter for the method if the test set is used
            if self.args.used_test_set == "test" and self.args.tunable:
                self.set_method_opt_param()

            # Run task
            logs = method.run_task(task_dic=tasks, shot=self.args.shots)

            if self.save_mult_outlier_distrib and hasattr(method, "mults"):
                if "support" in method.mults:
                    self.pred_mults["support"].append(method.mults["support"].detach())
                self.pred_mults["query"].append(method.mults["query"].detach())

            acc_mean, acc_conf = compute_confidence_interval(logs["acc"][:, -1])
            timestamps, criterions = logs["timestamps"], logs["criterions"]
            results_task.append(acc_mean)
            results_task_time.append(timestamps)
            for key in criterions.keys():
                results_criterion[key].append(criterions[key])

            del method
            del tasks

        if self.save_mult_outlier_distrib:
            if len(self.true_mults["support"]) > 0:
                self.true_mults["support"] = torch.cat(self.true_mults["support"])
                self.save_mult(
                    self.true_mults["support"], is_support=True, is_true=True
                )
                if len(self.pred_mults["support"]) > 0:
                    self.pred_mults["support"] = torch.cat(self.pred_mults["support"])
                    self.save_mult(
                        self.pred_mults["support"], is_support=True, is_true=False
                    )
            if len(self.true_mults["query"]) > 0:
                self.true_mults["query"] = torch.cat(self.true_mults["query"])
                self.save_mult(self.true_mults["query"], is_support=False, is_true=True)
                if len(self.pred_mults["query"]) > 0:
                    self.pred_mults["query"] = torch.cat(self.pred_mults["query"])
                    self.save_mult(
                        self.pred_mults["query"], is_support=True, is_true=False
                    )

        mean_accuracies = np.mean(np.asarray(results_task))
        mean_times = np.mean(np.asarray(results_task_time))
        mean_criterions = {}
        for key in results_criterion.keys():
            mean_criterions[key] = np.mean(results_criterion[key], axis=0)

        return mean_accuracies, mean_times, mean_criterions

    def get_method_val_param(self):
        # fixes for each method the name of the parameter on which validation is performed
        if self.args.name_method == "RPADDLE":
            self.val_param = self.args.lambd

    def report_results(self, mean_accuracies, mean_times):
        """
        Reports the results of the evaluation

        Args :
            mean_accuracies : Mean accuracies of the evaluation.
            mean_times : Mean time taken for evaluation.
        """
        self.logger.info("----- Final results -----")

        path = "results_few_shot/{}/{}".format(
            self.args.used_test_set, self.args.dataset
        )

        # If validation mode
        if self.args.used_test_set == "val":
            self.get_method_val_param()
            name_file = path + "/{}_s{}.txt".format(
                self.args.name_method, self.args.shots
            )

            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, "a")
            else:
                f = open(name_file, "w")
                f.write("val_param" + "\t" + "acc" + "\n")

            self.logger.info(
                "{}-shot mean test accuracy over {} tasks: {}".format(
                    self.args.shots, self.args.number_tasks, mean_accuracies
                )
            )

            f.write(str(self.val_param) + "\t")
            f.write(str(round(100 * mean_accuracies, 2)) + "\t")
            f.write("\n")
            f.close()

        # if test mode
        elif self.args.used_test_set == "test" and self.args.save_results == True:
            var = (
                str(self.args.shots)
                + "\t"
                + str(self.args.n_query)
                + "\t"
                + str(self.args.k_eff)
            )
            var_names = (
                "shots" + "\t" + "n_query" + "\t" + "k_eff" + "\t" + "acc" + "\n"
            )

            path = "results_few_shot/{}/{}".format(
                self.args.used_test_set, self.args.dataset
            )
            name_file = path + "/{}_s{}.txt".format(
                self.args.name_method, self.args.shots
            )

            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, "a")
            else:
                f = open(name_file, "w")
                f.write(var_names + "\t" + "\n")

            self.logger.info(
                "{}-shot mean test accuracy over {} tasks: {}".format(
                    self.args.shots, self.args.number_tasks, mean_accuracies
                )
            )
            self.logger.info(
                "{}-shot mean time over {} tasks: {}".format(
                    self.args.shots, self.args.number_tasks, mean_times
                )
            )
            f.write(str(var) + "\t")
            f.write(str(round(100 * mean_accuracies, 1)) + "\t")
            f.write("\n")
            f.close()

        else:
            self.logger.info(
                "{}-shot mean test accuracy over {} tasks: {}".format(
                    self.args.shots, self.args.number_tasks, mean_accuracies
                )
            )
            self.logger.info(
                "{}-shot mean time over {} tasks: {}".format(
                    self.args.shots, self.args.number_tasks, mean_times
                )
            )
