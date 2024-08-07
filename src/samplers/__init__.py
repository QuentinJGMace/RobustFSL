from .abstract_sampler import CategoriesSampler_few_shot
from .uniform_sampler import UniformSampler
from .balanced_sampler import BalancedSampler
from .dirichlet_sampler import Dirichlet_Sampler

SAMPLERS = {
    "uniform": UniformSampler,
    "balanced": BalancedSampler,
    "dirichlet": Dirichlet_Sampler,
}
