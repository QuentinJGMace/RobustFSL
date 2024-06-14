from src.api.cfg_utils import load_cfg_from_cfg_file


def load_cfg():
    cfg = load_cfg_from_cfg_file("config/main_config.yaml")
    dataset_config = "config/datasets_config/config_{}.yaml".format(cfg.dataset)
    method_config = "config/methods_config/{}.yaml".format(cfg.method)
    backbone_config = "config/backbones_config/{}.yaml".format(cfg.backbone)
    cfg.update(load_cfg_from_cfg_file(dataset_config))
    cfg.update(load_cfg_from_cfg_file(method_config))
    cfg.update(load_cfg_from_cfg_file(backbone_config))

    cfg.n_class = cfg.num_classes_test

    return cfg
