import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.nn.init import normal, constant
import math

class MindLSTMSep(nn.Module):
    def __init__(self):
        super(MindLSTMSep, self).__init__()
        self.feature_dim = 20
        self.seq_len = 5
        self.base_model = nn.Linear(self.feature_dim, self.feature_dim)

        self.lstm_m1 = nn.LSTM(self.feature_dim, self.feature_dim,bidirectional=False,num_layers=1,batch_first=True)
        self.lstm_m2 = nn.LSTM(self.feature_dim, self.feature_dim, bidirectional=False, num_layers=1, batch_first=True)
        self.lstm_m12 = nn.LSTM(self.feature_dim, self.feature_dim, bidirectional=False, num_layers=1, batch_first=True)
        self.lstm_m21 = nn.LSTM(self.feature_dim, self.feature_dim, bidirectional=False, num_layers=1, batch_first=True)
        self.lstm_mc = nn.LSTM(self.feature_dim, self.feature_dim, bidirectional=False, num_layers=1, batch_first=True)
        # The linear layer that maps the LSTM with the 3 outputs
        self.fc1_m1 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_m2 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_m12 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_m21 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_mc = nn.Linear(2 * self.feature_dim, 20)

        self.fc2_m1 = nn.Linear(20, 4)
        self.fc2_m2 = nn.Linear(20, 4)
        self.fc2_m12 = nn.Linear(20, 4)
        self.fc2_m21 = nn.Linear(20, 4)
        self.fc2_mc = nn.Linear(20, 4)

        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout()


    def forward(self, input):

        base_out = self.base_model(input.view((-1, input.size()[-1])))
        base_out = self.relu(base_out)
        base_out = base_out.view(input.size(0), self.seq_len, self.feature_dim)
        lstm_out_m1, _ = self.lstm_m1(base_out)
        lstm_out_m1 = self.dropout(lstm_out_m1)
        lstm_out_m1 = lstm_out_m1[:,4,:]
        lstm_out_m1 = torch.cat([lstm_out_m1, input[:, 4, :]], dim=1)

        lstm_out_m2, _ = self.lstm_m2(base_out)
        lstm_out_m2 = self.dropout(lstm_out_m2)
        lstm_out_m2 = lstm_out_m2[:, 4, :]
        lstm_out_m2 = torch.cat([lstm_out_m2, input[:, 4, :]], dim=1)

        lstm_out_m12, _ = self.lstm_m12(base_out)
        lstm_out_m12 = self.dropout(lstm_out_m12)
        lstm_out_m12 = lstm_out_m12[:, 4, :]
        lstm_out_m12 = torch.cat([lstm_out_m12, input[:, 4, :]], dim=1)

        lstm_out_m21, _ = self.lstm_m21(base_out)
        lstm_out_m21 = self.dropout(lstm_out_m21)
        lstm_out_m21 = lstm_out_m21[:, 4, :]
        lstm_out_m21 = torch.cat([lstm_out_m21, input[:, 4, :]], dim=1)

        lstm_out_mc, _ = self.lstm_mc(base_out)
        lstm_out_mc = self.dropout(lstm_out_mc)
        lstm_out_mc = lstm_out_mc[:, 4, :]
        lstm_out_mc = torch.cat([lstm_out_mc, input[:, 4, :]], dim=1)

        m1_out = self.fc1_m1(lstm_out_m1)
        # m1_out = self.dropout(m1_out)
        m1_out = self.relu(m1_out)
        m1_out = self.fc2_m1(m1_out)

        m2_out = self.fc1_m2(lstm_out_m2)
        # m2_out = self.dropout(m2_out)
        m2_out = self.relu(m2_out)
        m2_out = self.fc2_m2(m2_out)

        m12_out = self.fc1_m12(lstm_out_m12)
        # m12_out = self.dropout(m12_out)
        m12_out = self.relu(m12_out)
        m12_out = self.fc2_m12(m12_out)

        m21_out = self.fc1_m21(lstm_out_m21)
        # m21_out = self.dropout(m21_out)
        m21_out = self.relu(m21_out)
        m21_out = self.fc2_m21(m21_out)

        mc_out = self.fc1_mc(lstm_out_mc)
        # mc_out = self.dropout(mc_out)
        mc_out = self.relu(mc_out)
        mc_out = self.fc2_mc(mc_out)

        return m1_out, m2_out, m12_out, m21_out, mc_out

class MindLSTM(nn.Module):
    def __init__(self):
        super(MindLSTM, self).__init__()
        self.feature_dim = 20
        self.seq_len = 5
        self.base_model = nn.Linear(self.feature_dim, self.feature_dim)

        self.lstm = nn.LSTM(self.feature_dim, self.feature_dim,bidirectional=False,num_layers=1,batch_first=True)
        # The linear layer that maps the LSTM with the 3 outputs
        self.fc1_m1 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_m2 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_m12 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_m21 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_mc = nn.Linear(2 * self.feature_dim, 20)

        self.fc2_m1 = nn.Linear(20, 4)
        self.fc2_m2 = nn.Linear(20, 4)
        self.fc2_m12 = nn.Linear(20, 4)
        self.fc2_m21 = nn.Linear(20, 4)
        self.fc2_mc = nn.Linear(20, 4)

        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout()


    def forward(self, input):

        base_out = self.base_model(input.view((-1, input.size()[-1])))
        base_out = self.relu(base_out)
        base_out = base_out.view(input.size(0), self.seq_len, self.feature_dim)
        lstm_out, _ = self.lstm(base_out)
        # lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out[:,4,:]

        lstm_out = torch.cat([lstm_out, input[:, 4, :]], dim=1)

        m1_out = self.fc1_m1(lstm_out)
        m1_out = self.dropout(m1_out)
        m1_out = self.relu(m1_out)
        m1_out = self.fc2_m1(m1_out)

        m2_out = self.fc1_m2(lstm_out)
        m2_out = self.dropout(m2_out)
        m2_out = self.relu(m2_out)
        m2_out = self.fc2_m2(m2_out)

        m12_out = self.fc1_m12(lstm_out)
        m12_out = self.dropout(m12_out)
        m12_out = self.relu(m12_out)
        m12_out = self.fc2_m12(m12_out)

        m21_out = self.fc1_m21(lstm_out)
        m21_out = self.dropout(m21_out)
        m21_out = self.relu(m21_out)
        m21_out = self.fc2_m21(m21_out)

        mc_out = self.fc1_mc(lstm_out)
        mc_out = self.dropout(mc_out)
        mc_out = self.relu(mc_out)
        mc_out = self.fc2_mc(mc_out)

        return m1_out, m2_out, m12_out, m21_out, mc_out

class MindGRU(nn.Module):
    def __init__(self):
        super(MindGRU, self).__init__()
        self.feature_dim = 20
        self.seq_len = 5
        self.base_model = nn.Linear(self.feature_dim, self.feature_dim)

        self.gru = nn.GRU(self.feature_dim, self.feature_dim,bidirectional=False,num_layers=1,batch_first=True)
        # The linear layer that maps the LSTM with the 3 outputs
        self.fc1_m1 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_m2 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_m12 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_m21 = nn.Linear(2 * self.feature_dim, 20)
        self.fc1_mc = nn.Linear(2 * self.feature_dim, 20)

        self.fc2_m1 = nn.Linear(20, 4)
        self.fc2_m2 = nn.Linear(20, 4)
        self.fc2_m12 = nn.Linear(20, 4)
        self.fc2_m21 = nn.Linear(20, 4)
        self.fc2_mc = nn.Linear(20, 4)

        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout()


    def forward(self, input):

        base_out = self.base_model(input.view((-1, input.size()[-1])))
        base_out = self.relu(base_out)
        base_out = base_out.view(input.size(0), self.seq_len, self.feature_dim)
        gru_out, _ = self.gru(base_out)
        gru_out = gru_out[:,4,:]

        gru_out = torch.cat([gru_out, input[:, 4, :]], dim=1)

        m1_out = self.fc1_m1(gru_out)
        m1_out = self.dropout(m1_out)
        m1_out = self.relu(m1_out)
        m1_out = self.fc2_m1(m1_out)

        m2_out = self.fc1_m2(gru_out)
        m2_out = self.dropout(m2_out)
        m2_out = self.relu(m2_out)
        m2_out = self.fc2_m2(m2_out)

        m12_out = self.fc1_m12(gru_out)
        m12_out = self.dropout(m12_out)
        m12_out = self.relu(m12_out)
        m12_out = self.fc2_m12(m12_out)

        m21_out = self.fc1_m21(gru_out)
        m21_out = self.dropout(m21_out)
        m21_out = self.relu(m21_out)
        m21_out = self.fc2_m21(m21_out)

        mc_out = self.fc1_mc(gru_out)
        mc_out = self.dropout(mc_out)
        mc_out = self.relu(mc_out)
        mc_out = self.fc2_mc(mc_out)

        return m1_out, m2_out, m12_out, m21_out, mc_out

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(22, 20)
        self.fc_mc = torch.nn.Linear(20, 4)
        self.fc_m1 = torch.nn.Linear(20, 4)
        self.fc_m2 = torch.nn.Linear(20, 4)
        self.fc_m12 = torch.nn.Linear(20, 4)
        self.fc_m21 = torch.nn.Linear(20, 4)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        embedding = self.fc1(x)
        out = self.relu(embedding)
        out = self.dropout(out)
        out_mc = self.fc_mc(out)
        out_m1 = self.fc_m1(out)
        out_m2 = self.fc_m2(out)
        out_m12 = self.fc_m12(out)
        out_m21 = self.fc_m21(out)
        return out_mc, out_m12, out_m21,out_m1, out_m2