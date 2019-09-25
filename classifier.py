import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.conv2d(3, 32, 3, 1)
