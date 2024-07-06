import torch
from activation_funs.relu import relu_af


class CNNLayer():
    def __init__(self, stride, padding=None) -> None:
        super(CNNLayer, self).__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, input_mtx, kernel):
        
        output = relu_af(conv_output)

        return output




if __name__ == "__main__":
    cnnlayer = CNNLayer(2)
    output = cnnlayer(input_mtx, kernel)

    print(output)
