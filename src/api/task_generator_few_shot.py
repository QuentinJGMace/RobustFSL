import torch


class Task_Generator_Few_shot:
    def __init__(
        self,
        k_eff,
        shot,
        n_query,
        n_class,
        loader_support,
        loader_query,
        data_mean,
        outliers_dict,
        backbone,
        args,
    ):
        """
        Args:
            k_eff (int): Effective number of samples per class in support set.
            shot (int): the number of support samples per class.
            n_query (int): the number of query samples per class.
            n_class (int): the number of classes.
            loader_support (DataLoader): the data loader for the support set.
            loader_query (DataLoader): the data loader for the query set.
            data_mean (torch.tensor): the mean of the features of the training set.
            backbone (nn.Module): backbone used for feature extraction
            args (argparse.Namespace): the arguments.
        """

        self.k_eff = k_eff
        self.shot = shot
        self.n_query = n_query
        self.loader_support = loader_support
        self.loader_query = loader_query
        self.data_mean = data_mean
        self.outliers_dict = outliers_dict
        self.backbone = backbone
        self.args = args

    def get_task(
        self,
        data_support,
        data_query,
        labels_support,
        labels_query,
        outliers_support,
        outliers_query,
    ):
        """
        inputs:
            data_support : torch.tensor of shape [shot * k_eff, channels, H, W]
            data_query : torch.tensor of shape [n_query, channels, H, W]
            labels_support :  torch.tensor of shape [shot * k_eff + n_query]
            labels_query :  torch.tensor of shape [n_query]
            outliers_support : torch.tensor of shape [n_outliers_support]
            outliers_query : torch.tensor of shape [n_outliers_query]
        returns :
            task : Dictionary : x_support : torch.tensor of shape [k_eff * shot, channels, H, W]
                                x_query : torch.tensor of shape [n_query, channels, H, W]
                                y_support : torch.tensor of shape [k_eff * shot]
                                y_query : torch.tensor of shape [n_query]
                                x_mean: torch.tensor of shape [feature_dim]
                                outliers_support : torch.tensor of shape [n_outliers_support]
                                outliers_query : torch.tensor of shape [n_outliers_query]
        """

        unique_labels = torch.flip(
            torch.unique(labels_support, sorted=False), dims=(0,)
        )
        new_labels_support = torch.zeros_like(labels_support)
        new_labels_query = torch.zeros_like(labels_query)

        label_correspondance = {}

        for j, y in enumerate(unique_labels):
            new_labels_support[labels_support == y] = j
            new_labels_query[labels_query == y] = j
            label_correspondance[j] = y.int().item()

        new_data_query = data_query
        new_data_support = data_support

        torch.cuda.empty_cache()

        task = {
            "x_s": new_data_support,
            "y_s": new_labels_support.long(),
            "x_q": new_data_query,
            "y_q": new_labels_query.long(),
            "x_mean": self.data_mean,
            "outliers_support": outliers_support.long(),
            "outliers_query": outliers_query.long(),
            "label_correspondance": label_correspondance,
        }
        return task

    def generate_tasks(self):
        """
        returns :
            merged_task : { x_support : torch.tensor of shape [batch_size, k_eff * shot, channels, H, W]
                            x_query : torch.tensor of shape [batch_size, k_eff * query_shot, channels, H, W]
                            y_support : torch.tensor of shape [batch_size, k_eff * shot]
                            y_query : torch.tensor of shape [batch_size, k_eff * query_shot]
                            train_mean: torch.tensor of shape [feature_dim]}
        """

        tasks_dics = []

        for support, query, outliers_support, outliers_query in zip(
            self.loader_support,
            self.loader_query,
            self.outliers_dict["support"],
            self.outliers_dict["query"],
        ):
            (data_support, labels_support) = support
            (data_query, labels_query) = query
            task = self.get_task(
                data_support,
                data_query,
                labels_support,
                labels_query,
                outliers_support,
                outliers_query,
            )
            tasks_dics.append(task)

        feature_size = data_support.size()[-1]

        # Now merging all tasks into 1 single dictionary
        merged_tasks = {}
        n_task = len(tasks_dics)
        for key in tasks_dics[0].keys():
            if key in ["x_s", "x_q", "y_s", "y_q"]:
                n_samples = tasks_dics[0][key].size(0)
            if key == "x_s" or key == "x_q":
                merged_tasks[key] = torch.cat(
                    [tasks_dics[i][key] for i in range(n_task)], dim=0
                ).view(n_task, n_samples, feature_size)
            elif key == "y_s" or key == "y_q":
                merged_tasks[key] = torch.cat(
                    [tasks_dics[i][key] for i in range(n_task)], dim=0
                ).view(n_task, n_samples, -1)
            elif key == "x_mean":
                merged_tasks[key] = torch.cat(
                    [tasks_dics[i][key] for i in range(n_task)], dim=0
                ).view(n_task, -1)
            elif key == "outliers_support" or key == "outliers_query":
                merged_tasks[key] = torch.cat(
                    [tasks_dics[i][key] for i in range(n_task)], dim=0
                ).view(n_task, -1)
            elif key == "label_correspondance":
                merged_tasks[key] = {
                    k: [tasks_dics[i][key][k] for i in range(n_task)]
                    for k in tasks_dics[0][key].keys()
                }
            else:
                raise Exception("Wrong dict key")

        return merged_tasks
