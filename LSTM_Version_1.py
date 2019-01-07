import sys
import time

import torch.nn as nn

from HelperFunctions import *

# Using CUDA if we have a GPU that supports it along with the correct install, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Defining the network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.float()
        x = x.view(-1, self.sequence_length, self.input_size)

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def main():
    # Hyper-parameters
    epochs = 25
    batch_size = 32
    learning_rate = 0.0005
    input_size = 100
    hidden_size = 160
    num_layers = 2
    num_classes = 10
    sequence_length = 160

    #####################################################################################################
    # Initialize - model, criterion and optimizer
    model = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length).to(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #####################################################################################################
    # Initiate training and testing phases
    t0 = time.time()
    directory_name = "version_1_images"
    weights_file_name = None

    if len(sys.argv) == 2:
        weights_file_name = sys.argv[1]

    if weights_file_name is None:
        train(model, device, batch_size, epochs, criterion, optimizer, directory_name)
    else:
        model.load_state_dict(torch.load(weights_file_name))

    test(model, device, batch_size, directory_name)
    t1 = time.time()
    print("Time taken(in minutes): ", (t1 - t0) / 60)
    pass


if __name__ == '__main__':
    main()
