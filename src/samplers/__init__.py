from .abstract_sampler import CategoriesSampler_few_shot
from .uniform_sampler import UniformSampler
from .balanced_sampler import BalancedSampler

SAMPLERS = {
    "uniform": UniformSampler,
    "balanced": BalancedSampler,
}
