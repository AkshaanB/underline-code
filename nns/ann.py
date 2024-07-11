import os
import torch

from activation_funs.relu import relu_af

class ANNLayer():
    def __init__(self, weight, bias) -> None:
        super(ANNLayer, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        linear_output = self.weight * x + self.bias

        return relu_af(linear_output)
    

if __name__ == "__main__":
    ann_layer = ANNLayer(weight=torch.tensor(5, dtype=int), bias=torch.tensor(3, dtype=int))

    output = ann_layer.forward(torch.tensor(1, dtype=int))

    print(output)
