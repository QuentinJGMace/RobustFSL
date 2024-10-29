# RobustFSL

Repository used to implement and test RobustFSL methods. In particular the one developped during my intership generalizing the PADDLE algorithm (https://github.com/SegoleneMartin/PADDLE)

## Installation

- Clone this repo : `git clone https://github.com/QuentinJGMace/RobustFSL`
- Go to the root of the repo : `cd RobustFSL`
- install it : `pip install -e .`

## Usage

1) Download the backbones
    - run `python src/api/load_backbones_from_hub.py`
 This should download the backbones trained on miniimagenet used during my internship (they are publicly available here : https://huggingface.co/QuentinJG/ResNet18-miniimagenet and here : https://huggingface.co/QuentinJG/FeatResNet12-miniimagenet)

2) Download the datasets

    - MiniImagenet:

        You can download miniimagenet here : https://www.kaggle.com/datasets/arjunashok33/miniimagenet

        The splits are avalaible on in this repo inside `splits/miniimagenet` and should be moved to the root of the miniimagenet folder

    - Cifar10 : TODO

3) Extract the features
    - run `python src/api/extract_features.py`

4) Run FSL evaluation
    - run `python main.py`

    The configs can be modified in the configs folder

5) Run experiments
    - If one wants to run an experiment by varying some parameters, he can define a dictionary with the parameters he wants to change and then create an Experiement class

        An example is given at the end of the src/experiment/run_experiment.py file

## Logs and results
Main results from my work are all inside the `parse_experiments.ipynb` notebook, logs used to create those results are available at this link : TODO

## Repo structure

This repository is organised as follows, all source code is contained in the `src`folder which is subdivided in :
- **api** : high level functions, mainly : extracting features, evaluating few shot tasks, downloading backbones
- **backbones** : Code for the backbones architecture, loading and saving them
- **dataset** : Code handling the different datasets, data loaders and generating tasks
- **experiements** : Code to run experiments
- **logger** : Utility folder to log the results of runs
- **methods** : Where FSL algorithms are defined, all methods should inherit from the asbtract_method class
- **sampler** : Classes to define the query and support set samplers, mainly "uniform", "balanced" and "dirichlet" samplers over the query set

## Aknowledgment
The feat-resnet12 backbone is the one used in this repo "https://github.com/ebennequin/few-shot-open-set", thanks to their contributor for making it public.

 The resnet18 is the one used by Martin et al. in the NeurIPS 2022 paper "Towards Practical Few-shot Query Sets: Transductive Minimum Description Length Inference". Thanks to her for giving it to us.

 Thanks to Nora Ouzir and Jean-Christophe Pesquet for their help developping this algorithm