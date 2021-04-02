import numpy as np
import torch
from torch.autograd import Variable
import pickle
from tqdm import tqdm
import os
from sklearn import metrics

#data_path = '/home/shuwen/data/data_preprocessing2/mind_training/'
data_path = '/home/shuwen/data/data_preprocessing2/mind_retraining/'

clips = os.listdir(data_path)
data = []
labels = []
frame_records = []
for clip in clips:
    with open(data_path + clip, 'rb') as f:
        vec_input, label = pickle.load(f)
        data = data + vec_input
        labels = labels + label

data = np.array(data)
data = data.reshape((-1, data.shape[-1]))
labels = np.array(labels)
ratio = len(data)*0.8
ratio = int(ratio)
indx = np.random.randint(0, len(data), ratio)
flags = np.zeros(len(data))
flags[indx] = 1
train_x, train_y = data[flags == 1], labels[flags == 1]
test_x, test_y = data[flags == 0], labels[flags == 0]

def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy()/len(original)

def cal_corr(idx, actual_val):
    correct = 0
    idx = idx.cpu().numpy()
    # compare it with actual value and estimate accuracy
    for i in range(idx.shape[0]):
        if idx[i] == actual_val[i]:
            correct += 1
    return float(correct)/idx.shape[0]

def compute_acc(mc, m21, m12, m1, m2, mg, labels_):
    labels_mc = np.argmax(labels_[:, :4], axis=1)
    labels_m21 = np.argmax(labels_[:, 4:8], axis=1)
    labels_m12 = np.argmax(labels_[:, 8:12], axis=1)
    labels_m1 = np.argmax(labels_[:, 12:16], axis=1)
    labels_m2 = np.argmax(labels_[:, 16:20], axis=1)
    labels_mg = np.argmax(labels_[:, 20:], axis=1)
    max_score, idx_mc = torch.max(mc, 1)
    max_score, idx_m21 = torch.max(m21, 1)
    max_score, idx_m12 = torch.max(m12, 1)
    max_score, idx_m1 = torch.max(m1, 1)
    max_score, idx_m2 = torch.max(m2, 1)
    max_score, idx_mg = torch.max(mg, 1)
    acc_mc = cal_corr(idx_mc, labels_mc)
    acc_m12 = cal_corr(idx_m12, labels_m12)
    acc_m21 = cal_corr(idx_m21, labels_m21)
    acc_m1 = cal_corr(idx_m1, labels_m1)
    acc_m2 = cal_corr(idx_m2, labels_m2)
    acc_mg = cal_corr(idx_mg, labels_mg)
    return acc_mc, acc_m12, acc_m21, acc_m1, acc_m2,acc_mg

def test_score(net, data, label, batch_size):
    total = 0
    acc_total_mc, acc_total_m12, acc_total_m21, acc_total_m1, acc_total_m2, acc_total_mg = 0, 0, 0, 0, 0, 0
    net.eval()
    pbar = tqdm(range(0, data.shape[0], batch_size))
    total_mc = np.empty(0)
    total_m21 = np.empty(0)
    total_m12 = np.empty(0)
    total_m2 = np.empty(0)
    total_m1 = np.empty(0)
    total_mg = np.empty(0)
    total_act_mc = np.empty(0)
    total_act_m21 = np.empty(0)
    total_act_m12 = np.empty(0)
    total_act_m2 = np.empty(0)
    total_act_m1 = np.empty(0)
    total_act_mg = np.empty(0)
    for batch_num in pbar:
        total += 1
        if batch_num + batch_size > data.shape[0]:
            end = data.shape[0]
        else:
            end = batch_num + batch_size

        inputs_, actual_val = data[batch_num:end, :], label[batch_num:end]
        # perform classification
        inputs = torch.from_numpy(inputs_).float().cuda()
        # actual_val = torch.from_numpy(actual_val)
        mc, m21, m12, m1, m2, mg = net(torch.autograd.Variable(inputs))
        labels_mc = np.argmax(actual_val[:, :4], axis=1)
        labels_m21 = np.argmax(actual_val[:, 4:8], axis=1)
        labels_m12 = np.argmax(actual_val[:, 8:12], axis=1)
        labels_m1 = np.argmax(actual_val[:, 12:16], axis=1)
        labels_m2 = np.argmax(actual_val[:, 16:20], axis=1)
        labels_mg = np.argmax(actual_val[:, 20:], axis=1)
        max_score, idx_mc = torch.max(mc, 1)
        max_score, idx_m21 = torch.max(m21, 1)
        max_score, idx_m12 = torch.max(m12, 1)
        max_score, idx_m1 = torch.max(m1, 1)
        max_score, idx_m2 = torch.max(m2, 1)
        max_score, idx_mg = torch.max(mg, 1)
        total_mc = np.append(total_mc, idx_mc.cpu().numpy())
        total_m21 = np.append(total_m21, idx_m21.cpu().numpy())
        total_m12 = np.append(total_m12, idx_m12.cpu().numpy())
        total_m1 = np.append(total_m1, idx_m1.cpu().numpy())
        total_m2 = np.append(total_m2, idx_m2.cpu().numpy())
        total_mg = np.append(total_mg, idx_mg.cpu().numpy())

        total_act_mc = np.append(total_act_mc, labels_mc)
        total_act_m21 = np.append(total_act_m21, labels_m21)
        total_act_m12 = np.append(total_act_m12, labels_m12)
        total_act_m1 = np.append(total_act_m1, labels_m1)
        total_act_m2 = np.append(total_act_m2, labels_m2)
        total_act_mg = np.append(total_act_mg, labels_mg)
        pbar.set_description("processing batch %s" % str(batch_num))

    # print(set(total_act_mc) - set(total_mc))
    # print(set(total_act_m12) - set(total_m12))
    # print(set(total_act_m21) - set(total_m21))
    # print(set(total_act_m1) - set(total_m1))
    # print(set(total_act_m2) - set(total_m2))
    # print(set(total_act_mg) - set(total_mg))
    print(metrics.classification_report(total_act_mc, total_mc, digits=3))
    print(metrics.classification_report(total_act_m12, total_m12, digits=3))
    print(metrics.classification_report(total_act_m21, total_m21, digits=3))
    print(metrics.classification_report(total_act_m1, total_m1, digits=3))
    print(metrics.classification_report(total_act_m2, total_m2, digits=3))
    print(metrics.classification_report(total_act_mg, total_mg, digits=3))

def test(net, data, label, batch_size):
    total = 0
    acc_total_mc, acc_total_m12, acc_total_m21, acc_total_m1, acc_total_m2, acc_total_mg = 0, 0, 0, 0, 0, 0
    net.eval()
    pbar = tqdm(range(0, data.shape[0], batch_size))
    for batch_num in pbar:
        total += 1
        if batch_num + batch_size > data.shape[0]:
            end = data.shape[0]
        else:
            end = batch_num + batch_size

        inputs_, actual_val = data[batch_num:end, :], label[batch_num:end]
        # perform classification
        inputs = torch.from_numpy(inputs_).float().cuda()
        # actual_val = torch.from_numpy(actual_val)
        mc, m21, m12, m1, m2, mg = net(torch.autograd.Variable(inputs))
        acc_mc, acc_m12, acc_m21, acc_m1, acc_m2, acc_mg = compute_acc(mc, m21, m12, m1, m2, mg, actual_val)
        acc_total_mc += acc_mc
        acc_total_m12 += acc_m12
        acc_total_m21 += acc_m21
        acc_total_m1 += acc_m1
        acc_total_m2 += acc_m2
        acc_total_mg += acc_mg
        # convert 'predicted_val' GPU tensor to CPU tensor and extract the column with max_score
        # predicted_val = predicted_val.data.cpu()
        # for id, predicted_val_ in enumerate(predicted_val):
        #     print(torch.round(predicted_val_).eq(actual_val[id]).sum())
        #     print(torch.round(predicted_val_), actual_val[id])
        # accuracy += pred_acc(actual_val, predicted_val)
        pbar.set_description("processing batch %s" % str(batch_num))
    print("mc:{}, m1:{}, m2:{}, m12:{}, m21:{}, mg:{} ".format(acc_total_mc/total, acc_total_m1/total,
                                                               acc_total_m2/total, acc_total_m12/total,
                                                               acc_total_m21/total, acc_total_mg/total))

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(22, 22)
        self.fc_mc = torch.nn.Linear(22, 4)
        self.fc_m1 = torch.nn.Linear(22, 4)
        self.fc_m2 = torch.nn.Linear(22, 4)
        self.fc_m12 = torch.nn.Linear(22, 4)
        self.fc_m21 = torch.nn.Linear(22, 4)
        self.fc_mg = torch.nn.Linear(22, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        embedding = self.fc1(x)
        out = self.relu(embedding)
        out_mc = self.fc_mc(out)
        out_m1 = self.fc_m1(out)
        out_m2 = self.fc_m2(out)
        out_m12 = self.fc_m12(out)
        out_m21 = self.fc_m21(out)
        out_mg = self.fc_mg(out)
        return out_mc, out_m12, out_m21,out_m1, out_m2, out_mg

learningRate = 0.001
epochs = 100
batch_size = 32

def train():
    model = MLP()
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()
    weights = [1/39., 1/3., 1/391., 1/3702.]
    weights = torch.FloatTensor(weights).cuda()
    criterionc = torch.nn.CrossEntropyLoss(weight=weights)
    weights = [1/87., 1/11., 1/929., 1/3108.]
    weights = torch.FloatTensor(weights).cuda()
    criterionm12 = torch.nn.CrossEntropyLoss(weight=weights)
    weights = [1/78., 1/11., 1/847., 1/3199.]
    weights = torch.FloatTensor(weights).cuda()
    criterionm21 = torch.nn.CrossEntropyLoss(weight=weights)
    weights = [1/175., 1/6., 1/391., 1/3702.]
    weights = torch.FloatTensor(weights).cuda()
    criterionm1 = torch.nn.CrossEntropyLoss(weight=weights)
    weights = [1/177., 1/5., 1/2599., 1/1354.]
    weights = torch.FloatTensor(weights).cuda()
    criterionm2 = torch.nn.CrossEntropyLoss(weight=weights)
    criterionmg = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        # training set -- perform model training
        epoch_training_loss = 0.0
        num_batches = 0
        pbar = tqdm(range(0, train_x.shape[0], batch_size))
        for batch_num in pbar:  # 'enumerate' is a super helpful function
            # split training data into inputs and labels
            if batch_num+ batch_size> train_x.shape[0]:
                end = train_x.shape[0]
            else:
                end = batch_num+batch_size
            inputs_, labels_ =train_x[batch_num:end, :], train_y[batch_num:end]
            labels_mc = np.argmax(labels_[:, :4], axis=1)
            labels_m21 = np.argmax(labels_[:, 4:8], axis=1)
            labels_m12 = np.argmax(labels_[:, 8:12], axis=1)
            labels_m1 = np.argmax(labels_[:, 12:16], axis=1)
            labels_m2 = np.argmax(labels_[:, 16:20], axis=1)
            labels_mg = np.argmax(labels_[:, 20:], axis=1)
            inputs = torch.from_numpy(inputs_).float().cuda()
            labels = torch.from_numpy(labels_).cuda()
            labelsc = torch.from_numpy(labels_mc).cuda()
            labelsm12 = torch.from_numpy(labels_m12).cuda()
            labelsm21 = torch.from_numpy(labels_m21).cuda()
            labelsm1 = torch.from_numpy(labels_m1).cuda()
            labelsm2 = torch.from_numpy(labels_m2).cuda()
            labelsmg = torch.from_numpy(labels_mg).cuda()
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
            labelsc, labelsm12, labelsm21, labelsm1, labelsm2, labelsmg = torch.autograd.Variable(labelsc), \
            torch.autograd.Variable(labelsm12), torch.autograd.Variable(labelsm21), torch.autograd.Variable(labelsm1), \
            torch.autograd.Variable(labelsm2), torch.autograd.Variable(labelsmg)

            # Make gradients zero for parameters 'W', 'b'
            optimizer.zero_grad()
            mc, m21, m12, m1, m2, mg = model(inputs)
            loss = criterionc(mc, labelsc)
            loss += criterionm2(m2, labelsm2)
            loss += criterionm1(m1, labelsm1)
            loss += criterionm12(m12, labelsm12)
            loss += criterionm21(m21, labelsm21)
            loss += criterionmg(mg, labelsmg)
            loss.backward()

            optimizer.step()
            # calculating loss
            epoch_training_loss += loss.data.item()
            num_batches += 1
            # print(loss.data.item())
            pbar.set_description("processing batch %s" % str(batch_num))

        print("epoch: ", epoch, ", loss: ", epoch_training_loss / num_batches)
        if epoch == 99:
            test_score(model, test_x, test_y, batch_size=2000)
            break
        if epoch%10 == 0:
            save_path = './cptk/model_' +str(epoch)+'.pth'
            torch.save(model.state_dict(), save_path)

train()

def test_data(data, label, batch_size = 2000):
    net = MLP()
    net.load_state_dict(torch.load('./cptk/model_90.pth'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    pbar = tqdm(range(0, data.shape[0], batch_size))
    total = 0
    accuracy = 0
    for batch_num in pbar:
        total += 1
        if batch_num + batch_size > data.shape[0]:
            end = data.shape[0]
        else:
            end = batch_num + batch_size

        inputs_, actual_val = data[batch_num:end, :], label[batch_num:end]
        # perform classification
        inputs = torch.from_numpy(inputs_).float().cuda()
        actual_val = torch.from_numpy(actual_val)
        predicted_val = net(torch.autograd.Variable(inputs))
        # convert 'predicted_val' GPU tensor to CPU tensor and extract the column with max_score
        predicted_val = predicted_val.data.cpu()
        for id, predicted_val_ in enumerate(predicted_val):
            print(torch.round(predicted_val_).eq(actual_val[id]).sum())
            print(torch.round(predicted_val_), actual_val[id])
        accuracy += pred_acc(actual_val, predicted_val)
        pbar.set_description("processing batch %s" % str(batch_num))
    print("Classifier Acc: ", accuracy / total)

# test_data(data, labels)