import os
from tqdm import tqdm
import numpy as np
import torch

# import utils and datasets
from src.api.utils import Logger, save_pickle, load_pickle, compute_confidence_interval
from src.api.task_generator_few_shot import Task_Generator_Few_shot
from src.api.sampler_few_shot import (
    CategoriesSampler_few_shot,
    SamplerSupport_few_shot,
    SamplerQuery_few_shot,
)
from src.dataset import MiniImageNet
from src.dataset import build_data_loader
from src.methods import RobustPaddle

# Dataset list for FSL tasks
dataset_list = {
    "miniimagenet": MiniImageNet,
}


class Evaluator_few_shot:
    def __init__(self, device, args, log_file):
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, log_file)

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

    def extract_features(self, backbone, data_loader, set_name):
        """
        Extracts features for the given data loader and saves them.

        Args:
            backbone: The model to be evaluated.
            data_loader: Data loader for the dataset.
            set_name: Name of the set to be saved.
        Returns:
            None (saves the features on disk)
        """
        features = []
        labels = []

        all_features, all_labels = None, None

        for i, (data, target) in enumerate(tqdm(data_loader)):
            data = data.to(self.device)
            with torch.no_grad():
                features = backbone(data)
                features /= features.norm(dim=-1, keepdim=True)

                if i == 0:
                    all_features = features
                    all_labels = target.cpu()
                else:
                    all_features = torch.cat((all_features, features), dim=0)
                    all_labels = torch.cat((all_labels, target.cpu()), dim=0)

        try:
            os.mkdir(f"data/{self.args.dataset}/saved_features/")
        except:
            pass

        filepath = os.path.join(
            f"data/{self.args.dataset}/saved_features/{set_name}_features_{self.args.backbone}.pkl"
        )
        save_pickle(
            filepath,
            {
                "features": all_features,
                "labels": all_labels,
            },
        )

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
            self.extract_features(backbone, data_loaders["train"], "train")
        if not os.path.exists(
            f"data/{self.args.dataset}/saved_features/val_features_{self.args.backbone}.pkl"
        ):
            self.extract_features(backbone, data_loaders["val"], "val")
        if not os.path.exists(
            f"data/{self.args.dataset}/saved_features/test_features_{self.args.backbone}.pkl"
        ):
            self.extract_features(backbone, data_loaders["test"], "test")

        filepath_support = os.path.join(
            f"data/{self.args.dataset}/saved_features/{self.args.used_set_support}_features_{self.args.backbone}.pkl"
        )
        filepath_query = os.path.join(
            f"data/{self.args.dataset}/saved_features/{self.args.used_set_query}_features_{self.args.backbone}.pkl"
        )

        print(filepath_support)
        print(filepath_query)
        extracted_features_dic_support = load_pickle(filepath_support)
        extracted_features_dic_query = load_pickle(filepath_query)

        return extracted_features_dic_support, extracted_features_dic_query

    def run_full_evaluation(self, backbone, preprocess):
        """
        Run the full evaluation process over all tasks.
        :param backbone: The model to be evaluated.
        :param preprocess: Preprocessing function for data.
        :return: Mean accuracies of the evaluation.
        """

        # backbone.eval()

        # Init dataset and data loaders
        dataset = dataset_list[self.args.dataset](self.args.dataset_path)
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
        mean_accuracies, mean_times = self.evaluate_tasks(
            backbone, features_support, labels_support, features_query, labels_query
        )

        self.report_results(mean_accuracies, mean_times)

    def get_method_builder(self, model, device, args, log_file):
        # Initialize method classifier builder
        method_info = {
            "model": model,
            "device": device,
            "log_file": log_file,
            "args": args,
        }

        # few-shot methods
        if args.name_method == "RPADDLE":
            method_builder = RobustPaddle(**method_info)

        else:
            raise ValueError(
                "The method your entered does not exist or is not a few-shot method. Please check the spelling"
            )
        return method_builder

    def set_method_opt_param(self):
        """
        Set the optimal parameter for the method if the test set is used
        """
        raise NotImplementedError("Optimal parameter setting not implemented yet")

    def check_intersection(self, indices_support, indices_query):
        for i, indices in enumerate(indices_support):
            for indice in indices:
                if indice in indices_query[i]:
                    raise ValueError(
                        "Support and query sets have data points in common"
                    )

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

        results = []
        results_time = []
        results_task = []
        results_task_time = []

        # Evaluation over each task
        for i in range(int(self.args.number_tasks / self.args.batch_size)):

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

            sampler_support = SamplerSupport_few_shot(sampler)
            sampler_query = SamplerQuery_few_shot(sampler)

            # TODO : I think it is buggy,
            # the support and query sets could have data points in common
            # Get query and support samples
            test_loader_query = []
            indices_support, indices_query = [], []
            for indices in sampler_query:
                test_loader_query.append(
                    (features_query[indices, :], labels_query[indices])
                )
                indices_query.append(indices)

            test_loader_support = []
            for indices in sampler_support:
                test_loader_support.append(
                    (features_support[indices, :], labels_support[indices])
                )
                indices_support.append(indices)

            self.check_intersection(indices_support, indices_query)

        #     # Prepare the tasks
        #     task_generator = Task_Generator_Few_shot(
        #         k_eff=self.args.k_eff, shot=self.args.shots, n_query=self.args.n_query, n_class=self.args.n_class, loader_support=test_loader_support, loader_query=test_loader_query, backbone=backbone, args = self.args
        #     )

        #     tasks = task_generator.generate_tasks()

        #     # Load the method
        #     method = self.get_method_builder(
        #         backbone=backbone, device=self.device, args=self.args, log_file = self.log_file
        #     )
        #     # set the optimal parameter for the method if the test set is used
        #     if self.args.used_test_set == 'test' and self.args.tunable:
        #         self.set_method_opt_param()

        #     # Run task
        #     logs = method.run_task(task_dic=tasks, shot=self.args.shots)
        #     acc_mean, acc_conf = compute_confidence_interval(
        #         logs['acc'][:, -1]
        #     )
        #     timestamps, criterions = logs["timestamps"], logs["criterions"]
        #     results_task.append(acc_mean)
        #     results_task_time.append(timestamps)

        #     del method
        #     del tasks

        # mean_accuracies = np.mean(np.asarray(results_task))
        # mean_times = np.mean(np.asarray(results_task_time))

        return 1 / 0

        return mean_accuracies, mean_times

    # TODO : implement it
    def report_results(self, mean_accuracies, mean_times):
        """
        Reports the results of the evaluation

        Args :
            mean_accuracies : Mean accuracies of the evaluation.
            mean_times : Mean time taken for evaluation.
        """
        raise NotImplementedError("Report logs not implemented yet")
