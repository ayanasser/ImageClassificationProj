import torch.nn as nn


class TrafficSignClassifier(nn.Module):
    def __init__(self, input_size=12288, num_classes=4):
        super(TrafficSignClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 4096)
        self.batchnorm1 = nn.BatchNorm1d(4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, 100)
        self.batchnorm4 = nn.BatchNorm1d(100)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.batchnorm4(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        return x
