import torch
import torch.nn.functional as F


class CNNLayer():
    def __init__(self, stride, padding=None) -> None:
        super(CNNLayer, self).__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, input_mtx, kernel):

        if self.padding:
            padded_input_mtx = F.pad(input_mtx, self.padding, mode='constant', value=0)

            # Equation to get output size = (input_size - kernel_size + 2 * padding) / stride + 1
            output_mtx_height = input_mtx.shape[0]
            output_mtx_width = input_mtx.shape[1]

            kernel_height, kernel_width = kernel.shape

            output_mtx = torch.zeros((output_mtx_height, output_mtx_width), dtype=torch.int)

            for i in range(output_mtx_height):
                for j in range(output_mtx_width):
                    region = padded_input_mtx[i:i+kernel_height, j:j+kernel_width]
                    output_mtx[i,j] = torch.sum(region * kernel)

            conv_output = output_mtx
        else:
            # Equation to get output size = (input_size - kernel_size + 2 * padding) / stride + 1
            output_mtx_height = input_mtx.shape[0] - kernel.shape[0] + 1
            output_mtx_width = input_mtx.shape[1] - kernel.shape[1] + 1

            kernel_height, kernel_width = kernel.shape

            output_mtx = torch.zeros((output_mtx_height, output_mtx_width), dtype=torch.int)

            for i in range(output_mtx_height):
                for j in range(output_mtx_width):
                    region = input_mtx[i:i+kernel_height, j:j+kernel_width]
                    output_mtx[i,j] = torch.sum(region * kernel)

            conv_output = output_mtx

        output = torch.relu(conv_output)

        return output


if __name__ == "__main__":
    # cnnlayer = CNNLayer(1, padding=None)
    cnnlayer = CNNLayer(1, padding=(1, 1, 1, 1))
    input_mtx = torch.tensor([[-45, 12, 5, 17],
                [22, 10, -35, 6],
                [88, -26, 51, 19],
                [9, 77, 42, -3]], dtype=torch.int)
    kernel = torch.tensor(
                        [[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], dtype=torch.int)
    output = cnnlayer.forward(input_mtx, kernel)

    print(output)
