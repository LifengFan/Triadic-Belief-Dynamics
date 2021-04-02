import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

CLASS = ['SingleGaze', 'MutualGaze', 'AvertGaze', 'GazeFollow', 'JointAtt']


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)
        f.close()
        self.pad_dim = 50

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index]
        if len(sequence['data']) <= self.pad_dim:
            padded = sequence['data'] + [0 for _ in range(self.pad_dim - len(sequence['data']))]
        else:
            padded = sequence['data'][:self.pad_dim]

        if len(sequence['len']) <= self.pad_dim:
            padded_len = sequence['len'] + [0 for _ in range(self.pad_dim - len(sequence['len']))]
        else:
            padded_len = sequence['len'][:self.pad_dim]
        return {'label': torch.tensor(CLASS.index(sequence['label'])), 'data': torch.tensor(padded).float(), 'len': torch.tensor(padded_len).float()}


class EDNet(nn.Module):
    def __init__(self):
        super(EDNet, self).__init__()
        self.encoder_1 = nn.Linear(50, 50)
        self.encoder_2 = nn.Linear(50, 50)
        self.decoder1 = nn.Linear(100, 50)
        self.decoder2 = nn.Linear(50, 3)

    def forward(self, x_1, x_2):
        latent_1 = F.relu(self.encoder_1(x_1))
        latent_2 = F.relu(self.encoder_2(x_2))
        x = F.relu(self.decoder1(torch.cat((latent_1, latent_2), 1)))
        x =self.decoder2(x)
        return x


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc_1 = nn.Linear(100, 5)

    def forward(self, x_1, x_2):
        return self.fc_1(torch.cat((x_1, x_2), 1))
        # return self.fc_2(F.dropout(F.relu(self.fc_1(torch.cat((x_1, x_2), 1))), 0.8))


def get_metric_from_confmat(confmat):

    N=5

    recall=np.zeros(N)
    precision=np.zeros(N)
    F_score=np.zeros(N)

    correct_cnt=0.
    total_cnt=0.

    for i in range(N):

        recall[i]=confmat[i,i]/(np.sum(confmat[i,:])+1e-7)

        precision[i]=confmat[i,i]/(np.sum(confmat[:,i])+1e-7)

        F_score[i]=2*precision[i]*recall[i]/(precision[i]+recall[i]+1e-7)

        correct_cnt+=confmat[i,i]

        total_cnt+=np.sum(confmat[i,:])

    acc=correct_cnt/total_cnt

    print('===> Confusion Matrix for Event Label: \n {}'.format(confmat.astype(int)))

    print('===> Precision: \n  [SingleGaze]: {} % \n [MutualGaze]: {} % \n [GazeAversion]: {} % \n [GazeFollowing]: {} % \n [JointAtt]: {} % \n'
          .format(precision[0]*100, precision[1]*100, precision[2]*100, precision[3]*100, precision[4]*100))

    print('===> Recall: \n [SingleGaze]: {} % \n [MutualGaze]: {} % \n [GazeAversion]: {} % \n [GazeFollowing]: {} % \n [JointAtt]: {} % \n'
          .format(recall[0]*100, recall[1]*100, recall[2]*100, recall[3]*100, recall[4]*100))

    print('===> F score: \n [SingleGaze]: {} % \n [MutualGaze]: {} % \n [GazeAversion]: {} % \n [GazeFollowing]: {} % \n [JointAtt]: {} % \n'
          .format(F_score[0]*100, F_score[1]*100, F_score[2]*100, F_score[3]*100, F_score[4]*100))

    print('===> Accuracy: {} %'.format(acc*100))


with open('event_fine_tune_input.p', 'rb') as f:
    event_inputs, event_labels = pickle.load(f)



c = list(zip(event_inputs, event_labels))
random.shuffle(c)
event_inputs, event_labels = zip(*c)
ratio = len(event_inputs)*0.8
ratio = int(ratio)
train_x, train_y = event_inputs[:ratio], event_labels[:ratio]
test_x, test_y = event_inputs[ratio:], event_labels[ratio:]

criterion = nn.CrossEntropyLoss()
net = EDNet()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
batch_size = 128

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


total_acc_top1= AverageMeter()
total_acc_top2=AverageMeter()


for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(0, len(train_x), batch_size):
        # get the inputs
        inputs = event_inputs[i:i + batch_size]
        input1s, input2s = np.empty((0, 50)), np.empty((0, 50))
        ignore_input_ids = []
        record_input_ids = []
        for input_id, input in enumerate(inputs):
            input1, input2 = input[2]
            input1_pad = np.zeros((1, 50))
            input2_pad = np.zeros((1, 50))
            for i in range(len(input1)):
                input1_pad[0, i] = input1[i]
            for i in range(len(input2)):
                input2_pad[0, i] = input2[i]
            input1s = np.vstack([input1s, input1_pad])
            input2s = np.vstack([input2s, input2_pad])
            record_input_ids.append(input_id)
        input1s = torch.tensor(input1s).float().cuda()
        input2s = torch.tensor(input2s).float().cuda()

        label = train_y[i:i + batch_size]
        label = torch.tensor(label).float().cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(input1s, input2s)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

print('Finished Training')
correct = 0.0
correct_2 = 0.0
total = 0.0
confmat = np.zeros((5, 5))
with torch.no_grad():
    for i in range(0, len(train_x), batch_size):
        # get the inputs
        inputs = event_inputs[i:i + batch_size]
        input1s, input2s = np.empty((0, 50)), np.empty((0, 50))
        ignore_input_ids = []
        record_input_ids = []
        for input_id, input in enumerate(inputs):
            input1, input2 = input[2]
            input1_pad = np.zeros((1, 50))
            input2_pad = np.zeros((1, 50))
            for i in range(len(input1)):
                input1_pad[0, i] = input1[i]
            for i in range(len(input2)):
                input2_pad[0, i] = input2[i]
            input1s = np.vstack([input1s, input1_pad])
            input2s = np.vstack([input2s, input2_pad])
            record_input_ids.append(input_id)
        input1s = torch.tensor(input1s).float().cuda()
        input2s = torch.tensor(input2s).float().cuda()

        label = train_y[i:i + batch_size]
        label = torch.tensor(label).float().cuda()

        outputs = net(input1s, input2s)
        outputs.data = torch.rand((1, 5))
        _, predicted = torch.max(outputs.data, 1)
        valuse, ind = torch.topk(outputs.data, 2)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        for i in range(label.size(0)):
            confmat[predicted[i], label[i]] += 1
            if label[i] in ind.squeeze().numpy().tolist():
                correct_2 += 1
get_metric_from_confmat(confmat)


print('Top-1 Accuracy of the network on the test images: %f %%' % (100 * correct / total))

print('Top-2 Accuracy of the network on the test images: %f %%' % (100 * correct_2 / total))






