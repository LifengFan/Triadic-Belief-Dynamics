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

import units

class MindLSTMHog(torch.nn.Module):
    def __init__(self):
        super(MindLSTMHog, self).__init__()

        #--------------- base model ------------------
        self.resnet = units.Resnet.ResNet50(3)
        num_ftrs = self.resnet.resnet.fc.in_features
        self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 6)

        self.fc1=torch.nn.Linear(18,18)
        self.fc2=torch.nn.Linear(18,3)

        #-----------------------------------------------------
        # todo: load attmat weight
        # todo: freeze layer
        self._load_pretrained_weight('./model_attmat.pth')
        self.freeze_res_layer(layer_num=9)
        self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 54)

        self.reduce_dim1 = torch.nn.Linear(162 * 2, 100)
        self.reduce_dim2 = torch.nn.Linear(100, 60)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.bn2 = torch.nn.BatchNorm1d(60)
        self.bn3 = torch.nn.BatchNorm1d(270)
        self.bn4 = torch.nn.BatchNorm1d(36)

        self.feature_dim = 130
        self.fc1_m1 = nn.Linear(self.feature_dim, 20)
        self.fc1_m2 = nn.Linear(self.feature_dim, 20)
        self.fc1_m12 = nn.Linear(self.feature_dim, 20)
        self.fc1_m21 = nn.Linear(self.feature_dim, 20)
        self.fc1_mc = nn.Linear(self.feature_dim, 20)

        self.fc2_m1 = nn.Linear(20, 4)
        self.fc2_m2 = nn.Linear(20, 4)
        self.fc2_m12 = nn.Linear(20, 4)
        self.fc2_m21 = nn.Linear(20, 4)
        self.fc2_mc = nn.Linear(20, 4)

        self.relu  = nn.ReLU()

        self.lstm_m1 = nn.LSTM(self.feature_dim, self.feature_dim, bidirectional=False, num_layers=1, batch_first=True)
        self.lstm_m2 = nn.LSTM(self.feature_dim, self.feature_dim, bidirectional=False, num_layers=1, batch_first=True)
        self.lstm_m12 = nn.LSTM(self.feature_dim, self.feature_dim, bidirectional=False, num_layers=1, batch_first=True)
        self.lstm_m21 = nn.LSTM(self.feature_dim, self.feature_dim, bidirectional=False, num_layers=1, batch_first=True)
        self.lstm_mc = nn.LSTM(self.feature_dim, self.feature_dim, bidirectional=False, num_layers=1, batch_first=True)


    def _load_pretrained_weight(self, model_path):

        pretrained_model = torch.load(model_path)['state_dict']
        # ---------------------------------------------------------------------
        # load resnet weight
        model_dict=self.resnet.state_dict()
        pretrained_dict={}

        for k,v in pretrained_model.items():
            if k[len('module.resnet.'):] in model_dict:
                pretrained_dict[k[len('module.resnet.'):]]=v

        # print(len(model_dict))
        # print(len(pretrained_dict))

        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)
        # -----------------------------------------------------------------------
        # load fc1 weight
        pretrained_dict = {}
        model_dict=self.fc1.state_dict()
        for k,v in pretrained_model.items():
            #print('{} in pretrained'.format(k))
            if k.startswith('module.fc1.') and k[len('module.fc1.'):] in model_dict:
                #print('{} in model_dict'.format(k[len('module.fc1.'):]))
                pretrained_dict[k[len('module.fc1.'):]]=v

        model_dict.update(pretrained_dict)
        # self.fc1.load_state_dict(model_dict)
        # #-----------------------------------------------------------------------
        # # load fc2 weight
        # pretrained_dict = {}
        # model_dict=self.fc2.state_dict()
        # for k,v in pretrained_model.items():
        #     #print('{} in pretrained'.format(k))
        #     if k.startswith('module.fc2.') and k[len('module.fc2.'):] in model_dict:
        #         #print(k[len('module.fc2.'):])
        #         pretrained_dict[k[len('module.fc2.'):]]=v
        #
        # model_dict.update(pretrained_dict)
        # self.fc2.load_state_dict(model_dict)


    def freeze_res_layer(self, layer_num=9):

        # freeze resnet
        child_cnt=0
        for child in self.resnet.resnet.children():
            #print('-'*15)
            #print('resnet child {}'.format(child_cnt))
            #print(child)
            #if child_cnt<layer_num:
            for param in child.parameters():
                param.requires_grad=False

            child_cnt+=1

        print('Resnet has {} children totally, {} has been freezed'.format(child_cnt, child_cnt))

        # freeze fc
        # for param in self.fc1.parameters():
        #     param.requires_grad = False
        # for param in self.fc2.parameters():
        #     param.requires_grad = False


    def forward(self, event, obj_patch, hog_feature_input):

        obj_patch_input = obj_patch.view((-1, obj_patch.size()[-3], obj_patch.size()[-2], obj_patch.size()[-1]))
        nodes_feature = self.resnet(obj_patch_input)
        nodes_feature = nodes_feature.view(obj_patch.size()[0], obj_patch.size()[1], nodes_feature.size()[1])

        hog_feature = hog_feature_input.view((-1, hog_feature_input.size()[-1]))
        hog_feature = self.reduce_dim1(hog_feature)
        hog_feature = self.relu(hog_feature)
        hog_feature = self.bn1(hog_feature)
        hog_feature = self.reduce_dim2(hog_feature)
        hog_feature = self.relu(hog_feature)
        hog_feature = self.bn2(hog_feature)
        hog_feature = hog_feature.view(hog_feature_input.size()[0], hog_feature_input.size()[1], hog_feature.size()[1])

        fc_input = torch.cat([event, nodes_feature, hog_feature], 2)

        fc1_input, _ = self.lstm_m1(fc_input)
        fc1_input = fc1_input[:, 4, :]
        m1_out = self.fc1_m1(fc1_input)
        # m1_out = self.dropout(m1_out)
        m1_out = self.relu(m1_out)
        m1_out = self.fc2_m1(m1_out)

        fc2_input, _ = self.lstm_m2(fc_input)
        fc2_input = fc2_input[:, 4, :]
        m2_out = self.fc1_m2(fc2_input)
        # m2_out = self.dropout(m2_out)
        m2_out = self.relu(m2_out)
        m2_out = self.fc2_m2(m2_out)

        fc12_input, _ = self.lstm_m12(fc_input)
        fc12_input = fc12_input[:, 4, :]
        m12_out = self.fc1_m12(fc12_input)
        # m12_out = self.dropout(m12_out)
        m12_out = self.relu(m12_out)
        m12_out = self.fc2_m12(m12_out)

        fc21_input, _ = self.lstm_m21(fc_input)
        fc21_input = fc21_input[:, 4, :]
        m21_out = self.fc1_m21(fc21_input)
        # m21_out = self.dropout(m21_out)
        m21_out = self.relu(m21_out)
        m21_out = self.fc2_m21(m21_out)

        fcc_input, _ = self.lstm_mc(fc_input)
        fcc_input = fcc_input[:, 4, :]
        mc_out = self.fc1_mc(fcc_input)
        # mc_out = self.dropout(mc_out)
        mc_out = self.relu(mc_out)
        mc_out = self.fc2_mc(mc_out)

        return m1_out, m2_out, m12_out, m21_out, mc_out


class MindHog(torch.nn.Module):
    def __init__(self):
        super(MindHog, self).__init__()

        #--------------- base model ------------------
        self.resnet = units.Resnet.ResNet50(3)
        num_ftrs = self.resnet.resnet.fc.in_features
        self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 6)

        self.fc1=torch.nn.Linear(18,18)
        self.fc2=torch.nn.Linear(18,3)

        #-----------------------------------------------------
        # todo: load attmat weight
        # todo: freeze layer
        self._load_pretrained_weight('./model_attmat.pth')
        self.freeze_res_layer(layer_num=9)
        self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 54)

        self.reduce_dim1 = torch.nn.Linear(162 * 2, 100)
        self.reduce_dim2 = torch.nn.Linear(100, 60)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.bn2 = torch.nn.BatchNorm1d(60)
        self.bn3 = torch.nn.BatchNorm1d(270)
        self.bn4 = torch.nn.BatchNorm1d(36)

        self.feature_dim = 130
        self.fc1_m1 = nn.Linear(self.feature_dim, 20)
        self.fc1_m2 = nn.Linear(self.feature_dim, 20)
        self.fc1_m12 = nn.Linear(self.feature_dim, 20)
        self.fc1_m21 = nn.Linear(self.feature_dim, 20)
        self.fc1_mc = nn.Linear(self.feature_dim, 20)

        self.fc2_m1 = nn.Linear(20, 4)
        self.fc2_m2 = nn.Linear(20, 4)
        self.fc2_m12 = nn.Linear(20, 4)
        self.fc2_m21 = nn.Linear(20, 4)
        self.fc2_mc = nn.Linear(20, 4)

        self.relu  = nn.ReLU()


    def _load_pretrained_weight(self, model_path):

        pretrained_model = torch.load(model_path)['state_dict']
        # ---------------------------------------------------------------------
        # load resnet weight
        model_dict=self.resnet.state_dict()
        pretrained_dict={}

        for k,v in pretrained_model.items():
            if k[len('module.resnet.'):] in model_dict:
                pretrained_dict[k[len('module.resnet.'):]]=v

        # print(len(model_dict))
        # print(len(pretrained_dict))

        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)
        # -----------------------------------------------------------------------
        # load fc1 weight
        pretrained_dict = {}
        model_dict=self.fc1.state_dict()
        for k,v in pretrained_model.items():
            #print('{} in pretrained'.format(k))
            if k.startswith('module.fc1.') and k[len('module.fc1.'):] in model_dict:
                #print('{} in model_dict'.format(k[len('module.fc1.'):]))
                pretrained_dict[k[len('module.fc1.'):]]=v

        model_dict.update(pretrained_dict)
        # self.fc1.load_state_dict(model_dict)
        # #-----------------------------------------------------------------------
        # # load fc2 weight
        # pretrained_dict = {}
        # model_dict=self.fc2.state_dict()
        # for k,v in pretrained_model.items():
        #     #print('{} in pretrained'.format(k))
        #     if k.startswith('module.fc2.') and k[len('module.fc2.'):] in model_dict:
        #         #print(k[len('module.fc2.'):])
        #         pretrained_dict[k[len('module.fc2.'):]]=v
        #
        # model_dict.update(pretrained_dict)
        # self.fc2.load_state_dict(model_dict)


    def freeze_res_layer(self, layer_num=9):

        # freeze resnet
        child_cnt=0
        for child in self.resnet.resnet.children():
            #print('-'*15)
            #print('resnet child {}'.format(child_cnt))
            #print(child)
            #if child_cnt<layer_num:
            for param in child.parameters():
                param.requires_grad=False

            child_cnt+=1

        print('Resnet has {} children totally, {} has been freezed'.format(child_cnt, child_cnt))

        # freeze fc
        # for param in self.fc1.parameters():
        #     param.requires_grad = False
        # for param in self.fc2.parameters():
        #     param.requires_grad = False


    def forward(self, event, obj_patch, hog_feature):

        nodes_feature = self.resnet(obj_patch)

        hog_feature = self.reduce_dim1(hog_feature)
        hog_feature = self.relu(hog_feature)
        hog_feature = self.bn1(hog_feature)
        hog_feature = self.reduce_dim2(hog_feature)
        hog_feature = self.relu(hog_feature)
        hog_feature = self.bn2(hog_feature)

        fc_input = torch.cat([event, nodes_feature, hog_feature], 1)

        m1_out = self.fc1_m1(fc_input)
        # m1_out = self.dropout(m1_out)
        m1_out = self.relu(m1_out)
        m1_out = self.fc2_m1(m1_out)

        m2_out = self.fc1_m2(fc_input)
        # m2_out = self.dropout(m2_out)
        m2_out = self.relu(m2_out)
        m2_out = self.fc2_m2(m2_out)

        m12_out = self.fc1_m12(fc_input)
        # m12_out = self.dropout(m12_out)
        m12_out = self.relu(m12_out)
        m12_out = self.fc2_m12(m12_out)

        m21_out = self.fc1_m21(fc_input)
        # m21_out = self.dropout(m21_out)
        m21_out = self.relu(m21_out)
        m21_out = self.fc2_m21(m21_out)

        mc_out = self.fc1_mc(fc_input)
        # mc_out = self.dropout(mc_out)
        mc_out = self.relu(mc_out)
        mc_out = self.fc2_mc(mc_out)

        return m1_out, m2_out, m12_out, m21_out, mc_out


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

class MLP_SEP(torch.nn.Module):
    def __init__(self):
        super(MLP_SEP, self).__init__()
        self.fc_event = torch.nn.Linear(50, 20)
        self.fc_event_2 = torch.nn.Linear(20, 10)
        self.fc_memory = torch.nn.Linear(50, 20)
        self.fc_memory_2 = torch.nn.Linear(20, 10)
        self.fc_mc = torch.nn.Linear(20, 4)
        self.fc_m1 = torch.nn.Linear(20, 4)
        self.fc_m2 = torch.nn.Linear(20, 4)
        self.fc_m12 = torch.nn.Linear(20, 4)
        self.fc_m21 = torch.nn.Linear(20, 4)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x1 = x[:, :, :10].reshape((x.size()[0], -1))
        x2 = x[:, :, 10:].reshape((x.size()[0], -1))

        x1 = self.relu(self.fc_event(x1))
        x2 = self.relu(self.fc_memory(x2))
        x1 = self.relu(self.fc_event_2(x1))
        x2 = self.relu(self.fc_memory_2(x2))
        embedding = torch.cat([x1, x2], 1)
        out = self.relu(embedding)
        out = self.dropout(out)
        out_mc = self.fc_mc(out)
        out_m1 = self.fc_m1(out)
        out_m2 = self.fc_m2(out)
        out_m12 = self.fc_m12(out)
        out_m21 = self.fc_m21(out)
        return out_mc, out_m12, out_m21,out_m1, out_m2

class MLP_Event_Memory(torch.nn.Module):
    def __init__(self):
        super(MLP_Event_Memory, self).__init__()
        self.fc_event = torch.nn.Linear(16, 16)
        self.fc_event1 = torch.nn.Linear(16, 10)
        self.fc1_mc = torch.nn.Linear(10, 10)
        self.fc1_m1 = torch.nn.Linear(10, 10)
        self.fc1_m2 = torch.nn.Linear(10, 10)
        self.fc1_m12 = torch.nn.Linear(10, 10)
        self.fc1_m21 = torch.nn.Linear(10, 10)
        self.fc_mc = torch.nn.Linear(10, 4)
        self.fc_m1 = torch.nn.Linear(10, 4)
        self.fc_m2 = torch.nn.Linear(10, 4)
        self.fc_m12 = torch.nn.Linear(10, 4)
        self.fc_m21 = torch.nn.Linear(10, 4)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout()
    def forward(self, x):
        x1 = self.relu(self.fc_event(x))
        out = self.relu(self.fc_event1(x1))
        out_mc = self.relu(self.fc1_mc(out))
        out_mc = self.fc_mc(out_mc)
        out_m1 = self.relu(self.fc1_m1(out))
        out_m1 = self.fc_m1(out_m1)
        out_m2 = self.relu(self.fc1_m2(out))
        out_m2 = self.fc_m2(out_m2)
        out_m12 = self.relu(self.fc1_m12(out))
        out_m12 = self.relu(self.fc_m12(out_m12))
        out_m21 = self.relu(self.fc1_m21(out))
        out_m21 = self.fc_m21(out_m21)
        return out_m1, out_m2, out_m12,out_m21, out_mc