import torch
import sys
import units
import torch.nn as nn

class AttMat(torch.nn.Module):
    def __init__(self):
        super(AttMat, self).__init__()

        #--------------- base model ------------------
        self.resnet = units.Resnet.ResNet50(3)
        num_ftrs = self.resnet.resnet.fc.in_features
        self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 6)

        self.fc1=torch.nn.Linear(18,18)
        self.fc2=torch.nn.Linear(18,3)

        #-----------------------------------------------------
        # todo: load attmat weight
        # todo: freeze layer
        self._load_pretrained_weight('./pretrained/model_attmat.pth')
        self.freeze_res_layer(layer_num=9)
        self.resnet.resnet.fc = torch.nn.Linear(num_ftrs, 54)

        self.reduce_dim1 = torch.nn.Linear(162 * 2, 100)
        self.reduce_dim2 = torch.nn.Linear(100, 60)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.bn2 = torch.nn.BatchNorm1d(60)
        self.bn3 = torch.nn.BatchNorm1d(270)
        self.bn4 = torch.nn.BatchNorm1d(36)

        self.feature_dim = 130 - 16
        self.fc1_m1 = nn.Linear(self.feature_dim, 20)

        self.fc2_m1 = nn.Linear(20, 3)

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


    def forward(self, obj_patch, hog_feature):

        nodes_feature = self.resnet(obj_patch)

        hog_feature = self.reduce_dim1(hog_feature)
        hog_feature = self.relu(hog_feature)
        # hog_feature = self.bn1(hog_feature)
        hog_feature = self.reduce_dim2(hog_feature)
        hog_feature = self.relu(hog_feature)
        # hog_feature = self.bn2(hog_feature)

        fc_input = torch.cat([nodes_feature, hog_feature], 1)

        m1_out = self.fc1_m1(fc_input)
        m1_out = self.relu(m1_out)
        m1_out = self.fc2_m1(m1_out)


        return torch.sigmoid(m1_out)