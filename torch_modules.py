import torch

class ScalarMult(torch.nn.Linear):
    def __init__(self, in_features: int, scalar: float):
        super(ScalarMult, self).__init__(in_features, in_features, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.eye(in_features) * scalar)

class ScalarAdd(torch.nn.Linear):
    def __init__(self, in_features: int, scalar: float):
        super(ScalarAdd, self).__init__(in_features, in_features)
        with torch.no_grad():
            self.weight.copy_(torch.eye(in_features))
            self.bias.fill_(scalar)

class SumVector(torch.nn.Linear):
    def __init__(self, in_features: int):
        super(SumVector, self).__init__(in_features, 1, bias=False)
        with torch.no_grad():
            self.weight.fill_(1.0)
