import torch.nn as nn
import torch.nn.init as init
class DQNetworkCNN(nn.Module):
    # Create a feedforward neural network. Class contains two functions:
    # init is used for setting architecture of the neural network
    # forward is used for calculating output of the nn given an input
    def __init__(self, output_size, input_size, hidden_size, device):
        super(DQNetworkCNN, self).__init__()  # inherrent values from parrent class.
        self.__device = device  # cuda or cpu
        self.__input_size = input_size
        self.__output_size = output_size

        self.__hidden_size = hidden_size

        self.__c1_block = nn.Sequential(
            nn.Conv2d(self.__input_size, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2,stride=2) # takes te maximum value,
            # used to reduce the number of parameters learned and helps avoid overfitting by providing a abstracted form.
        )

        FC_size = 3136  # The input size to the fully connected network

        self.__FC_block = nn.Sequential(
            # nn.Flatten(), #Flatten into single vector
            nn.Linear(in_features=FC_size, out_features=self.__hidden_size),
            nn.ReLU(),
            # nn.Linear(in_features=self.__hidden_size,  out_features=self.__hidden_size),
            # nn.ReLU(),
            nn.Linear(in_features=self.__hidden_size, out_features=self.__output_size)
        )

        # Initialize linear layers with kaiming initialization
        for layer in self.__FC_block:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        # Initialize conv2d layers with kaiming initialization
        for layer in self.__c1_block:
            if isinstance(layer, nn.Conv2d):
                init.kaiming_uniform_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        # with profiler.record_function("LINEAR PASS"):
        x = self.__c1_block(x)
        # with profiler.record_function("Conv PASS"):
        x = self.__FC_block(x.view(x.size(0), -1))
        return x

    def get_device(self):
        return self.__device

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size

    def get_hidden_size(self):
        return self.__hidden_size

class DQNetwork(nn.Module):
    def __init__(self, n_actions, n_inputs,device):
        super().__init__()
        hidden = 64
        self.device = device
        self.layer1 = nn.Linear(n_inputs, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, hidden)
        self.adv = nn.Linear(hidden, n_actions)

        self.l1_activation = nn.ReLU()
        self.l2_activation = nn.ReLU()
        self.l3_activation = nn.ReLU()

    def forward(self, x):
        y1 = self.l1_activation(self.layer1(x))
        y1 = self.l2_activation(self.layer2(y1))
        y1 = self.l3_activation(self.layer3(y1))
        adv = self.adv(y1)
        return adv
