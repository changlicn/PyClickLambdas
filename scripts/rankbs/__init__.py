from .bandits import CascadeUCB1
from .bandits import CascadeKL_UCB
from .bandits import CascadeThompsonSampler
from .bandits import CascadeLambdaMachine
from .bandits import CascadeExp3

try:
    __version__ = __import__('pkg_resources').get_distribution('rankbs').version
except:
    __version__ = '?'

__all__ = [
    'CascadeUCB1',
    'CascadeKL_UCB',
    'CascadeThompsonSampler',
    'CascadeLambdaMachine',
    'CascadeExp3',
]