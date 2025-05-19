import torch.nn as nn
import torch
import torch.nn.init as init

class DQNModel(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, device):
        super(DQNModel, self).__init__()
        self.__device = device
        self.__input_size = input_size
        self.__output_size = output_size
        self.__hidden_size = hidden_size

        # define the fully connected layers
        self.__c1_block = nn.Sequential(
            nn.Conv2d(self.__input_size, 32, kernel_size=2),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=2),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.LeakyReLU()
        ).to(device)

        # calculate conv output size
        dummy_input = torch.zeros(1, self.__input_size, 11, 11, device=self.__device)
        dummy_output = self.__c1_block(dummy_input)
        fc_size = dummy_output.numel()

        # define the fully connected layers
        self.__fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=fc_size, out_features=self.__hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.__hidden_size, out_features=self.__output_size)
        ).to(device)

        # initialize the layers
        for layer in self.__fc_block:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
        for layer in self.__c1_block:
            if isinstance(layer, nn.Conv2d):
                init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
        print(self.get_num_parameters())

    def forward(self, x):
        x = self.__c1_block(x)
        x = self.__fc_block(x)
        return x

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_device(self):
        return self.__device

    def get_output_size(self):
        return self.__output_size

    def get_hidden_size(self):
        return self.__hidden_size

    def get_input_size(self):
        return self.__input_size


class DQNetworkCNN(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, device):
        super(DQNetworkCNN, self).__init__() 
        self.__device = device
        self.__input_size = input_size
        self.__output_size = output_size
        self.__hidden_size = hidden_size

        self.__c1_block = nn.Sequential(
            nn.Conv2d(self.__input_size, 32, kernel_size=3),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.LeakyReLU()
        )

        dummy_input = torch.zeros(1, self.__input_size, 11, 11)
        dummy_output = self.__c1_block(dummy_input)
        FC_size = int(dummy_output.numel())

        self.__FC_block = nn.Sequential(
            nn.Linear(in_features=FC_size, out_features=self.__hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.__hidden_size, out_features=self.__output_size)
        )

        for layer in self.__FC_block:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
        for layer in self.__c1_block:
            if isinstance(layer, nn.Conv2d):
                init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
        print(self.get_num_parameters())

    def forward(self, x):
        x = self.__c1_block(x)
        x = self.__FC_block(x.view(x.size(0), -1))
        return x

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self):
        return self.__device
    
    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size

    def get_hidden_size(self):
        return self.__hidden_size
