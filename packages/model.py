import torch
import torch.nn as nn
import torch.optim as optim

class Face_classifier_model(nn.Module):
    def __init__(self):
        super(Face_classifier_model, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = Face_classifier_model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
