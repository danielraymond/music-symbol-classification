import torch
import torch.nn as nn
import torch.nn.functional as F


class NNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


####################################################


class NNClassifierWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NNClassifierWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        return self.fc4(x)


####################################################


class FinalNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FinalNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(hidden_dim, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.25)

        self.fc5 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        return self.fc5(x)


#################################################
# Original Author: Jake Snell
# Source: https://github.com/jakesnell/prototypical-networks
# The following code implements a prototypical network slightly edited for use with the self-supervised CNN feature encode.


class PrototypicalNetwork(nn.Module):
    def __init__(self, input_dim):
        super(PrototypicalNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, support, query, n_way, k_shot):
        support_embeddings = self.fc(support)  # (n_way * k_shot, input_dim)
        query_embeddings = self.fc(query)  # (num_query_samples, input_dim)

        prototypes = support_embeddings.view(n_way, k_shot, -1).mean(dim=1)

        dists = euclidean_dist(query_embeddings, prototypes)

        return dists


def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class PrototypicalLoss(nn.Module):
    def __init__(self):
        super(PrototypicalLoss, self).__init__()

    def forward(self, dists, target, n_way):
        log_p_y = F.log_softmax(-dists, dim=1)
        loss = -log_p_y.gather(1, target.view(-1, 1)).mean()
        _, y_hat = log_p_y.max(1)
        acc = (y_hat == target).float().mean()
        return loss, acc


#################################################


class FinalPrototypicalNetwork(nn.Module):
    def __init__(self, input_dim):
        super(FinalPrototypicalNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, support, query, n_way, k_shot):
        support_embeddings = self.fc1(support)
        support_embeddings = self.relu(support_embeddings)
        support_embeddings = self.bn(support_embeddings)
        support_embeddings = self.dropout(support_embeddings)

        query_embeddings = self.fc1(query)
        query_embeddings = self.relu(query_embeddings)
        query_embeddings = self.bn(query_embeddings)
        query_embeddings = self.dropout(query_embeddings)

        prototypes = support_embeddings.view(n_way, k_shot, -1).mean(dim=1)

        dists = euclidean_dist(query_embeddings, prototypes)

        return dists
