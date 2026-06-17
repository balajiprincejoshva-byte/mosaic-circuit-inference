# Expose core APIs
from .matrix_io import MultiOmicTensor
from .factor_graph import FactorGraphBP
from .rbm_thermo import MultiOmicRBM
from .perturbation import PerturbationSimulator
from .dynamics import LangevinSimulator
from .inverse_design import TargetOptimizer
from .cohort_sim import VirtualCohort

__all__ = [
    'MultiOmicTensor',
    'FactorGraphBP',
    'MultiOmicRBM',
    'PerturbationSimulator',
    'LangevinSimulator',
    'TargetOptimizer',
    'VirtualCohort'
]
