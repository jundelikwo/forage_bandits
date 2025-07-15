from .egree import EpsilonGreedy
from .ucb import UCB
from .ts import ThompsonSampling
from .discounteducb import DiscountedUCB
from .discountedegree import DiscountedEpsilonGreedy
from .discountedts import DiscountedThompsonSampling
from .neural_hebbian import NeuralHebbianAgent

__all__ = ['EpsilonGreedy', 'UCB', 'ThompsonSampling', 'DiscountedUCB', 'DiscountedEpsilonGreedy', 'DiscountedThompsonSampling', 'NeuralHebbianAgent'] 