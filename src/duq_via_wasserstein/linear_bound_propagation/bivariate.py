import bound_propagation as bp

__all__ = ['BoundVectorAdd']


class BoundVectorAdd(bp.BoundVectorAdd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def strict_ibp_forward(self, bounds, intersection, save_relaxation=False, save_input_bounds=False):
        bounds = self.ibp_forward(bounds, save_relaxation, save_input_bounds)
        intersection = self.module(intersection)

        return bounds, intersection
