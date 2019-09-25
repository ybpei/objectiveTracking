import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.relu = torch.ReLU()
