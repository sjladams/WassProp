from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Union, List
import torch
import discretize_distributions.distributions as dd_dists

TensorLike = Union[float, torch.Tensor]


@dataclass
class W2: # TODO make more informative
    p__q_empirical: TensorLike = 0.0
    p__q_global_lipschitz: TensorLike = 0.0
    p__q_lagrangian_duality: TensorLike = 0.0

@dataclass
class Distributions:
    q: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal, dd_dists.CategoricalFloat]
    noise: Optional[Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal]] = None
    q_emp: Optional[torch.Tensor] = None
    p_emp: Optional[torch.Tensor] = None

@dataclass
class SingleStepData:   # just contains single ambuigity set?
    w2: W2
    distributions: Distributions

@dataclass
class MultiStepData:
    steps: Dict[int, SingleStepData] = field(default_factory=dict)

    def append(self, k: int, rec: SingleStepData) -> None:
        self.steps[k] = rec

    def at(self, k: int) -> SingleStepData:
        return self.steps[k]

    def _ordered_k(self) -> List[int]:
        return sorted(self.steps.keys())

    def stack_w2(self, field: str) -> torch.Tensor:
        """Return a [K] or [K, B] tensor stacking a given W2 field over time."""
        vals = []
        for k in self._ordered_k():
            v = getattr(self.steps[k].w2, field)
            t = torch.as_tensor(v)
            t = torch.atleast_1d(t)
            vals.append(t)
        return torch.stack(vals, dim=0)

    def collect_distributions(self, field: str):
        """
        Collect a field (e.g. 'q1_emp') from all distributions.
        Returns a tensor if shapes match, else a list.
        """
        tensors, shapes = [], set()
        for k in self._ordered_k():
            x = getattr(self.steps[k].distributions, field)
            if x is None:
                return [getattr(self.steps[k].distributions, field) for k in self._ordered_k()]
            tensors.append(x)
            shapes.add(tuple(x.shape))
        if len(shapes) == 1:
            return torch.stack(tensors, dim=0)
        return [getattr(self.steps[k].distributions, field) for k in self._ordered_k()]