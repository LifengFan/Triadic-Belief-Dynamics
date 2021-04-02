import numpy as np
import torch
from torch.autograd import Variable
import pickle

from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import os
from sklearn import metrics
import glob
from annotation_clean import *
from sklearn import svm
import matplotlib.pyplot as plt
import joblib
from mind_model import *
import seaborn as sn

#data_path = '/home/shuwen/data/data_preprocessing2/mind_training/'
data_path = '/home/shuwen/data/data_preprocessing2/mind_lstm_training/'
model_type = 'lstm'

clips = os.listdir(data_path)
data = []
labels = []
frame_records = []
sep_data = {0:[], 1:[], 2:[], 3:[], 4:[]}
sep_label = {0:[], 1:[], 2:[], 3:[], 4:[]}
for clip in clips:
    with open(data_path + clip, 'rb') as f:
        vec_input, label_ = pickle.load(f)
        data = data + vec_input
        labels = labels + label_
        for lid, label in enumerate(label_):
            label = label.reshape(-1)
            if label[0] == 1 or label[4] == 1 or label[8] == 1 or label[12] == 1 or label[16] == 1:
                sep_data[0].append(vec_input[lid])
                sep_label[0].append(label)
            if label[1] == 1 or label[5] == 1 or label[9] ==  1 or label[13] == 1 or label[17] == 1:
                sep_data[1].append(vec_input[lid])
                sep_label[1].append(label)
            if label[2] == 1 or label[6] == 1 or label[10] == 1 or label[14] == 1 or label[18] == 1:
                sep_data[2].append(vec_input[lid])
                sep_label[2].append(label)
            if label[3] == 1 or label[7] == 1 or label[11] == 1 or label[15] == 1 or label[19] == 1:
                sep_data[3].append(vec_input[lid])
                sep_label[3].append(label)

if model_type == 'single':
    test_xs, test_ys = np.empty((0, 22)), np.empty((0, 21))  # single
else:
    test_xs, test_ys = np.empty((0, 5, 20)), np.empty((0, 20)) #seq

sep_train_x, sep_train_y = {}, {}
for i in sep_data.keys():
    if i == 4:
        continue
    data = sep_data[i]
    data = np.array(data)
    if model_type == 'single':
        data = data.reshape((-1, data.shape[-1]))
    label = sep_label[i]
    label = np.array(label)
    ratio = len(data) * 0.8
    ratio = int(ratio)
    indx = np.random.randint(0, len(data), ratio)
    flags = np.zeros(len(data))
    flags[indx] = 1
    train_x, train_y = data[flags == 1], label[flags == 1]
    test_x, test_y = data[flags == 0], label[flags == 0]
    sep_train_x[i] = train_x
    sep_train_y[i] = train_y

    test_xs = np.vstack([test_xs, test_x])
    test_ys = np.vstack([test_ys, test_y])

test_x, test_y = test_xs, test_ys
print(test_x.shape, test_y.shape)

max_sample = max(len(sep_train_x[0]), len(sep_train_x[1]), len(sep_train_x[2]), len(sep_train_x[3]))
if model_type == 'single':
    train_x, train_y = np.empty((0, 22)), np.empty((0, 21))  # single
else:
    train_x, train_y = np.empty((0, 5, 20)), np.empty((0, 20)) #seq

for i in sep_train_x.keys():
    repeat_time = max_sample/len(sep_train_x[i])
    # if i == 3:
    #     train_x = np.vstack([train_x, sep_train_x[i]])
    #     train_y = np.vstack([train_y, sep_train_y[i]])
    if not i==3:
        if model_type == 'single':
            train_x = np.vstack([train_x, np.tile(sep_train_x[i], (repeat_time*5, 1))])
        else:
            train_x = np.vstack([train_x, np.tile(sep_train_x[i], (repeat_time * 5, 1, 1))])
        train_y = np.vstack([train_y, np.tile(sep_train_y[i], (repeat_time*5, 1))])

if model_type == 'single':
    train_total = np.hstack([train_x, train_y])
else:
    train_total = np.hstack([train_x.reshape((-1, train_x.shape[1]*train_x.shape[2])), train_y])
np.random.shuffle(train_total)
train_size = int(train_total.shape[0]*0.75)

if model_type == 'single':
    train_x, train_y = train_total[:train_size, :22], train_total[:train_size, 22:]
    validate_x, validate_y = train_total[train_size:, :22], train_total[train_size:, 22:]
else:
    train_x, train_y = train_total[:train_size, :train_x.shape[1]*train_x.shape[2]].reshape((-1, train_x.shape[1], train_x.shape[2])), \
                       train_total[:train_size, train_x.shape[1]*train_x.shape[2]:]
    validate_x, validate_y = train_total[train_size:, :train_x.shape[1]*train_x.shape[2]].reshape((-1, train_x.shape[1], train_x.shape[2])), \
                             train_total[train_size:, train_x.shape[1]*train_x.shape[2]:]

# data = np.array(data)
# data = data.reshape((-1, data.shape[-1]))
# labels = np.array(labels)
# ratio = len(data)*0.8
# ratio = int(ratio)
# indx = np.random.randint(0, len(data), ratio)
# flags = np.zeros(len(data))
# flags[indx] = 1
# train_x, train_y = data[flags == 1], labels[flags == 1]
# test_x, test_y = data[flags == 0], labels[flags == 0]

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

def test_score(net, data, label, batch_size, proj_name, dataset):
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
        # mc, m21, m12, m1, m2 = net(torch.autograd.Variable(inputs))
        m1, m2, m12, m21, mc = net(torch.autograd.Variable(inputs))
        labels_mc = np.argmax(actual_val[:, :4], axis=1)
        labels_m21 = np.argmax(actual_val[:, 4:8], axis=1)
        labels_m12 = np.argmax(actual_val[:, 8:12], axis=1)
        labels_m1 = np.argmax(actual_val[:, 12:16], axis=1)
        labels_m2 = np.argmax(actual_val[:, 16:20], axis=1)
        # labels_mg = np.argmax(actual_val[:, 20:], axis=1)
        max_score, idx_mc = torch.max(mc, 1)
        max_score, idx_m21 = torch.max(m21, 1)
        max_score, idx_m12 = torch.max(m12, 1)
        max_score, idx_m1 = torch.max(m1, 1)
        max_score, idx_m2 = torch.max(m2, 1)
        # max_score, idx_mg = torch.max(mg, 1)
        total_mc = np.append(total_mc, idx_mc.cpu().numpy())
        total_m21 = np.append(total_m21, idx_m21.cpu().numpy())
        total_m12 = np.append(total_m12, idx_m12.cpu().numpy())
        total_m1 = np.append(total_m1, idx_m1.cpu().numpy())
        total_m2 = np.append(total_m2, idx_m2.cpu().numpy())
        # total_mg = np.append(total_mg, idx_mg.cpu().numpy())

        total_act_mc = np.append(total_act_mc, labels_mc)
        total_act_m21 = np.append(total_act_m21, labels_m21)
        total_act_m12 = np.append(total_act_m12, labels_m12)
        total_act_m1 = np.append(total_act_m1, labels_m1)
        total_act_m2 = np.append(total_act_m2, labels_m2)
        # total_act_mg = np.append(total_act_mg, labels_mg)
        pbar.set_description("processing batch %s" % str(batch_num))

    # print(set(total_act_mc) - set(total_mc))
    # print(set(total_act_m12) - set(total_m12))
    # print(set(total_act_m21) - set(total_m21))
    # print(set(total_act_m1) - set(total_m1))
    # print(set(total_act_m2) - set(total_m2))
    # print(set(total_act_mg) - set(total_mg))
    results_mc = metrics.classification_report(total_act_mc, total_mc, digits=3)
    results_m1 = metrics.classification_report(total_act_m1, total_m1, digits=3)
    results_m2 = metrics.classification_report(total_act_m2, total_m2, digits=3)
    results_m12 = metrics.classification_report(total_act_m12, total_m12, digits=3)
    results_m21 = metrics.classification_report(total_act_m21, total_m21, digits=3)

    print(results_mc)
    print(results_m1)
    print(results_m2)
    print(results_m12)
    print(results_m21)

    with open('./cptk_' + proj_name + '/' + dataset + '.p', 'wb') as f:
        pickle.dump([results_m1, results_m2, results_m12, results_m21, results_mc], f)



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


learningRate = 0.001
epochs = 1501
batch_size = 32

def train_svm():
    labels_mc = np.argmax(train_y[:, :4], axis=1)
    labels_m21 = np.argmax(train_y[:, 4:8], axis=1)
    labels_m12 = np.argmax(train_y[:, 8:12], axis=1)
    labels_m1 = np.argmax(train_y[:, 12:16], axis=1)
    labels_m2 = np.argmax(train_y[:, 16:20], axis=1)
    # labels_mg = np.argmax(train_y[:, 20:], axis=1)
    clf1 = svm.SVC(decision_function_shape='ovo')
    clf2 = svm.SVC(decision_function_shape='ovo')
    clf12 = svm.SVC(decision_function_shape='ovo')
    clf21 = svm.SVC(decision_function_shape='ovo')
    clfc = svm.SVC(decision_function_shape='ovo')
    clfg = svm.SVC(decision_function_shape='ovo')
    clf1.fit(train_x, labels_m1)
    clf2.fit(train_x, labels_m2)
    clf12.fit(train_x, labels_m12)
    clf21.fit(train_x, labels_m21)
    clfc.fit(train_x, labels_mc)
    # clfg.fit(train_x, labels_mg)
    pred_y1 = clf1.predict(validate_x)
    pred_y2 = clf2.predict(validate_x)
    pred_y12 = clf12.predict(validate_x)
    pred_y21 = clf21.predict(validate_x)
    pred_yc = clfc.predict(validate_x)
    # pred_yg = clfg.predict(validate_x)
    labels_mc_v = np.argmax(validate_y[:, :4], axis=1)
    labels_m21_v = np.argmax(validate_y[:, 4:8], axis=1)
    labels_m12_v = np.argmax(validate_y[:, 8:12], axis=1)
    labels_m1_v = np.argmax(validate_y[:, 12:16], axis=1)
    labels_m2_v = np.argmax(validate_y[:, 16:20], axis=1)
    # labels_mg_v = np.argmax(validate_y[:, 20:], axis=1)
    print(metrics.classification_report(labels_mc_v, pred_yc, digits=3))
    print(metrics.classification_report(labels_m12_v, pred_y12, digits=3))
    print(metrics.classification_report(labels_m21_v, pred_y21, digits=3))
    print(metrics.classification_report(labels_m1_v, pred_y1, digits=3))
    print(metrics.classification_report(labels_m2_v, pred_y2, digits=3))
    # print(metrics.classification_report(labels_mg_v, pred_yg, digits=3))
    save_path = './cptk_svm/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + 'clf_svm.p', 'wb') as f:
        joblib.dump([clf1, clf2, clf12, clf21, clfc], f)

def test_svm(test_x, test_y, proj_name, dataset):
    with open('./cptk_svm/clf_svm.p', 'rb') as f:
        clf1, clf2, clf12, clf21, clfc = joblib.load(f)

    # clfg.fit(train_x, labels_mg)
    pred_y1 = clf1.predict(test_x)
    pred_y2 = clf2.predict(test_x)
    pred_y12 = clf12.predict(test_x)
    pred_y21 = clf21.predict(test_x)
    pred_yc = clfc.predict(test_x)
    # pred_yg = clfg.predict(validate_x)
    labels_mc_v = np.argmax(test_y[:, :4], axis=1)
    labels_m21_v = np.argmax(test_y[:, 4:8], axis=1)
    labels_m12_v = np.argmax(test_y[:, 8:12], axis=1)
    labels_m1_v = np.argmax(test_y[:, 12:16], axis=1)
    labels_m2_v = np.argmax(test_y[:, 16:20], axis=1)
    # labels_mg_v = np.argmax(validate_y[:, 20:], axis=1)

    results_mc = metrics.classification_report(labels_mc_v, pred_yc, digits=3)
    results_m1 = metrics.classification_report(labels_m1_v, pred_y1, digits=3)
    results_m2 = metrics.classification_report(labels_m2_v, pred_y2, digits=3)
    results_m12 = metrics.classification_report(labels_m12_v, pred_y12, digits=3)
    results_m21 = metrics.classification_report(labels_m21_v, pred_y21, digits=3)

    print(results_mc)
    print(results_m1)
    print(results_m2)
    print(results_m12)
    print(results_m21)

    with open('./cptk_' + proj_name + '/' + dataset + '.p', 'wb') as f:
        pickle.dump([results_m1, results_m2, results_m12, results_m21, results_mc], f)


def train(save_prefix, checkpoint = None, startepoch = None):
    if save_prefix == 'lstm':
        model = MindLSTM()
    elif save_prefix == 'lstm_bidi':
        model = MindLSTM()
    elif save_prefix == 'gru':
        model = MindGRU()
    elif save_prefix == 'lstm_sep':
        model = MindLSTMSep()
    else:
        model = MLP()
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    if startepoch is not None:
        startepoch = startepoch
    else:
        startepoch = 0
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
    # criterionmg = torch.nn.CrossEntropyLoss()

    # criterionc = torch.nn.CrossEntropyLoss()
    # criterionm12 = torch.nn.CrossEntropyLoss()
    # criterionm21 = torch.nn.CrossEntropyLoss()
    # criterionm1 = torch.nn.CrossEntropyLoss()
    # criterionm2 = torch.nn.CrossEntropyLoss()
    # criterionmg = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    # scheduler = MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.1)
    losses_m1, losses_m2, losses_m12, losses_m21, losses_mc = [], [], [], [], []
    for epoch in range(startepoch, epochs):
        # training set -- perform model training
        epoch_training_loss_m1, epoch_training_loss_m2, epoch_training_loss_m12, epoch_training_loss_m21, epoch_training_loss_mc\
            = 0.0, 0.0, 0.0, 0.0, 0.0
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
            # labels_mg = np.argmax(labels_[:, 20:], axis=1)
            inputs = torch.from_numpy(inputs_).float().cuda()
            labels = torch.from_numpy(labels_).cuda()
            labelsc = torch.from_numpy(labels_mc).cuda()
            labelsm12 = torch.from_numpy(labels_m12).cuda()
            labelsm21 = torch.from_numpy(labels_m21).cuda()
            labelsm1 = torch.from_numpy(labels_m1).cuda()
            labelsm2 = torch.from_numpy(labels_m2).cuda()
            # labelsmg = torch.from_numpy(labels_mg).cuda()
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
            labelsc, labelsm12, labelsm21, labelsm1, labelsm2= torch.autograd.Variable(labelsc), \
            torch.autograd.Variable(labelsm12), torch.autograd.Variable(labelsm21), torch.autograd.Variable(labelsm1), \
            torch.autograd.Variable(labelsm2)

            # Make gradients zero for parameters 'W', 'b'
            optimizer.zero_grad()
            # mc, m21, m12, m1, m2, mg = model(inputs)
            m1, m2, m12, m21, mc = model(inputs)
            loss_m1, loss_m2, loss_m12, loss_m21, loss_mc = criterionm1(m1, labelsm1), criterionm2(m2, labelsm2), \
            criterionm12(m12, labelsm12), criterionm21(m21, labelsm21), criterionc(mc, labelsc)
            loss = loss_m1 + loss_m2 + loss_m12 + loss_m21 + loss_mc
            loss.backward()

            optimizer.step()
            # calculating loss
            epoch_training_loss_m1 += loss_m1.data.item()
            epoch_training_loss_m2 += loss_m2.data.item()
            epoch_training_loss_m12 += loss_m12.data.item()
            epoch_training_loss_m21 += loss_m21.data.item()
            epoch_training_loss_mc += loss_mc.data.item()
            num_batches += 1
            # print(loss.data.item())
            pbar.set_description("processing batch %s" % str(batch_num))
        # scheduler.step()
        print("epoch:{}/m1_loss:{}, m2_loss:{}, m12_loss:{}, m21_loss:{}, mc_loss:{}".format(
            epoch, epoch_training_loss_m1/num_batches, epoch_training_loss_m2/num_batches, epoch_training_loss_m12/num_batches,
            epoch_training_loss_m21 / num_batches, epoch_training_loss_mc/num_batches))
        losses_m1.append(epoch_training_loss_m1/num_batches)
        losses_m2.append(epoch_training_loss_m2 / num_batches)
        losses_m12.append(epoch_training_loss_m12 / num_batches)
        losses_m21.append(epoch_training_loss_m21 / num_batches)
        losses_mc.append(epoch_training_loss_mc / num_batches)
        # if epoch == 5:
        #     test_score(model, train_x, train_y, batch_size=2000)
        #     plt.plot(losses)
        #     plt.show()
        if epoch%50 == 0:
            save_path = './cptk_' + save_prefix + '/model_' +str(epoch)+'.pth'
            torch.save(model.state_dict(), save_path)
            legend_labels = ['m1', 'm2', 'm12', 'm21', 'mc']
            plt.plot(losses_m1)
            plt.plot(losses_m2)
            plt.plot(losses_m12)
            plt.plot(losses_m21)
            plt.plot(losses_mc)
            plt.legend(legend_labels)
            plt.show()

def check_overlap_return_area(head_box, obj_curr):
    max_left = max(head_box[0], obj_curr[0])
    max_top = max(head_box[1], obj_curr[1])
    min_right = min(head_box[2], obj_curr[2])
    min_bottom = min(head_box[3], obj_curr[3])
    if (min_right - max_left) > 0 and (min_bottom - max_top) > 0:
        return (min_right - max_left)*(min_bottom - max_top)
    return -100

def get_obj_name(obj_bbox, annt, frame_id):
    obj_candidates = annt.loc[annt.frame == frame_id]
    max_overlap = 0
    max_name = None
    max_bbox = None
    obj_bbox = [obj_bbox[0], obj_bbox[1], obj_bbox[0] + obj_bbox[2], obj_bbox[1] + obj_bbox[3]]
    obj_area = (obj_bbox[2] - obj_bbox[0])*(obj_bbox[3] - obj_bbox[1])
    for index, obj_candidate in obj_candidates.iterrows():
        if obj_candidate['name'].startswith('P'):
            continue
        candidate_bbox = [obj_candidate['x_min'], obj_candidate['y_min'], obj_candidate['x_max'], obj_candidate['y_max']]
        overlap = check_overlap_return_area(obj_bbox, candidate_bbox)
        if overlap > max_overlap and overlap/obj_area < 1.2 and overlap/obj_area > 0.8:
            max_overlap = overlap
            max_name = obj_candidate['name']
            max_bbox = candidate_bbox
    if max_overlap > 0:
        return max_name, max_bbox
    return None, None

def update_memory(memory, mind_name, fluent, loc):
    if fluent == 0 or fluent == 2:
        memory[mind_name]['loc'] = loc
    elif fluent == 1:
        memory[mind_name]['loc'] = None

    return memory

def get_grid_location_using_bbox(obj_frame):
    x_min = obj_frame[0]
    y_min = obj_frame[1]
    x_max = obj_frame[0] + obj_frame[2]
    y_max = obj_frame[1] + obj_frame[3]
    gridLW = 1280 / 25.
    gridLH = 720 / 15.
    center_x, center_y = (x_min + x_max)/2, (y_min + y_max)/2
    X, Y = int(center_x / gridLW), int(center_y / gridLH)
    return X, Y

def test_raw_data():
    # net = MindLSTM()
    # net.load_state_dict(torch.load('./cptk_lstm/model_200.pth'))
    # if torch.cuda.is_available():
    #     net.cuda()
    # net.eval()
    with open('./cptk_svm/clf_svm.p', 'rb') as f:
        clf1, clf2, clf12, clf21, clfc = joblib.load(f)

    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    obj_bbox_path = '/home/shuwen/data/data_preprocessing2/interpolate_bbox/'
    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    clips = os.listdir(event_label_path)
    m1_predict, m1_real = [], []
    m2_predict, m2_real = [], []
    m12_predict, m12_real = [], []
    m21_predict, m21_real = [], []
    mc_predict, mc_real = [], []
    mg_predict, mg_real = [], []
    for clip in clips[:10]:
        if not os.path.exists(reannotation_path + clip):
            continue
        # if not clip == 'test_94342_20.p':
        #     continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs, len(img_names))
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = glob.glob(obj_bbox_path + clip.split('.')[0] + '/*.p')
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        for obj_name in obj_names:
            print(obj_name)
            with open(obj_name, 'rb') as f:
                obj_bboxs = pickle.load(f)
            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            for frame_id in range(p1_events_by_frame.shape[0]):
                # curr_loc
                obj_bbox = obj_bboxs[frame_id]
                curr_loc = get_grid_location_using_bbox(obj_bbox)

                # gt
                gt_obj_name, gt_bbox = get_obj_name(obj_bbox, annt, frame_id)
                img = cv2.imread(img_names[frame_id])

                if not gt_obj_name:
                    continue

                # cv2.rectangle(img, (int(obj_bbox[0]), int(obj_bbox[1])), (int(obj_bbox[2] + obj_bbox[0]), int(obj_bbox[3] + obj_bbox[1])), (255, 0, 0), thickness=2)
                # cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 0, 255), thickness=2)
                # cv2.imshow('img', img)
                # cv2.waitKey(200)
                obj_record = obj_records[gt_obj_name][frame_id]
                for mind_name in obj_record:
                    if mind_name == 'mg':
                        continue
                    if mind_name == 'm1':
                        m1_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm2':
                        m2_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm12':
                        m12_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm21':
                        m21_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mc':
                        mc_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mg':
                        mg_real.append(obj_record[mind_name]['fluent'])

                # event
                p1_event = p1_events_by_frame[frame_id]
                p2_event = p2_events_by_frame[frame_id]

                # memory
                memory_dist = []
                indicator = []
                for mind_name in memory.keys():
                    if frame_id == 0:
                        memory_dist.append(0)
                        indicator.append(0)
                    else:
                        memory_loc = memory[mind_name]['loc']
                        if memory_loc is not None:
                            curr_loc = np.array(curr_loc)
                            memory_loc = np.array(memory_loc)
                            memory_dist.append(np.linalg.norm(curr_loc - memory_loc))
                            indicator.append(1)
                        else:
                            memory_dist.append(0)
                            indicator.append(0)
                # get predicted value
                memory_dist = np.array(memory_dist)
                indicator = np.array(indicator)
                input = np.hstack([p1_event.reshape((1, -1)), p2_event.reshape((1, -1)), memory_dist.reshape((1, -1)),
                                   indicator.reshape((1, -1))])
                idx_m1 = clf1.predict(input)
                idx_m2 = clf2.predict(input)
                idx_m12 = clf12.predict(input)
                idx_m21 = clf21.predict(input)
                idx_mc = clfc.predict(input)
                m1_predict.append(idx_m1[0])
                m2_predict.append(idx_m2[0])
                m12_predict.append(idx_m12[0])
                m21_predict.append(idx_m21[0])
                mc_predict.append(idx_mc[0])

                # inputs = torch.from_numpy(input).float().cuda()
                # mc, m21, m12, m1, m2, mg = net(inputs)
                # max_score, idx_mc = torch.max(mc, 1)
                # max_score, idx_m21 = torch.max(m21, 1)
                # max_score, idx_m12 = torch.max(m12, 1)
                # max_score, idx_m1 = torch.max(m1, 1)
                # max_score, idx_m2 = torch.max(m2, 1)
                # max_score, idx_mg = torch.max(mg, 1)
                #
                # m1_predict.append(idx_m1.cpu().numpy()[0])
                # m2_predict.append(idx_m2.cpu().numpy()[0])
                # m12_predict.append(idx_m12.cpu().numpy()[0])
                # m21_predict.append(idx_m21.cpu().numpy()[0])
                # mc_predict.append(idx_mc.cpu().numpy()[0])
                # mg_predict.append(idx_mg.cpu().numpy()[0])
                # update memory
                memory = update_memory(memory, 'm1', idx_m1, curr_loc)
                memory = update_memory(memory, 'm2', idx_m2, curr_loc)
                memory = update_memory(memory, 'm12', idx_m12, curr_loc)
                memory = update_memory(memory, 'm21', idx_m21, curr_loc)
                memory = update_memory(memory, 'mc', idx_mc, curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)

    print(metrics.classification_report(mc_real, mc_predict, digits=3))
    print(metrics.classification_report(m12_real, m12_predict, digits=3))
    print(metrics.classification_report(m21_real, m21_predict, digits=3))
    print(metrics.classification_report(m1_real, m1_predict, digits=3))
    print(metrics.classification_report(m2_real, m2_predict, digits=3))


def test_data_not_predict_svm():
    with open('./cptk_svm/clf_svm.p', 'rb') as f:
        clf1, clf2, clf12, clf21, clfc = joblib.load(f)
    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    obj_bbox_path = '/home/shuwen/data/data_preprocessing2/interpolate_bbox/'
    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    clips = os.listdir(event_label_path)
    m1_predict, m1_real = [], []
    m2_predict, m2_real = [], []
    m12_predict, m12_real = [], []
    m21_predict, m21_real = [], []
    mc_predict, mc_real = [], []
    mg_predict, mg_real = [], []
    for clip in clips[:10]:
        if not os.path.exists(reannotation_path + clip):
            continue
        if not clip == 'test_94342_20.p':
            continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs, len(img_names))
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = glob.glob(obj_bbox_path + clip.split('.')[0] + '/*.p')
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        for obj_name in obj_names:
            print(obj_name)
            with open(obj_name, 'rb') as f:
                obj_bboxs = pickle.load(f)
            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            for frame_id in range(p1_events_by_frame.shape[0]):

                # curr_loc
                obj_bbox = obj_bboxs[frame_id]
                curr_loc = get_grid_location_using_bbox(obj_bbox)
                # gt
                gt_obj_name, gt_bbox = get_obj_name(obj_bbox, annt, frame_id)
                img = cv2.imread(img_names[frame_id])
                if not gt_obj_name:
                    continue
                # cv2.rectangle(img, (int(obj_bbox[0]), int(obj_bbox[1])), (int(obj_bbox[2] + obj_bbox[0]), int(obj_bbox[3] + obj_bbox[1])), (255, 0, 0), thickness=2)
                # cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 0, 255), thickness=2)
                # cv2.imshow('img', img)
                # cv2.waitKey(200)

                obj_record = obj_records[gt_obj_name][frame_id]
                flag = 0
                for mind_name in obj_record:
                    if not obj_record[mind_name]['fluent'] == 3:
                        flag = 1
                if flag == 0:
                    continue
                for mind_name in obj_record:
                    if mind_name == 'm1':
                        m1_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm2':
                        m2_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm12':
                        m12_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm21':
                        m21_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mc':
                        mc_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mg':
                        mg_real.append(obj_record[mind_name]['fluent'])
                # event
                p1_event = p1_events_by_frame[frame_id]
                p2_event = p2_events_by_frame[frame_id]
                # memory
                memory_dist = []
                indicator = []
                for mind_name in memory.keys():
                    if frame_id == 0:
                        memory_dist.append(0)
                        indicator.append(0)
                    else:
                        memory_loc = obj_records[gt_obj_name][frame_id][mind_name]['loc']
                        if memory_loc is not None:
                            curr_loc = np.array(curr_loc)
                            memory_loc = np.array(memory_loc)
                            memory_dist.append(np.linalg.norm(curr_loc - memory_loc))
                            indicator.append(1)
                        else:
                            memory_dist.append(0)
                            indicator.append(0)
                # get predicted value
                memory_dist = np.array(memory_dist)
                indicator = np.array(indicator)
                input = np.hstack([p1_event.reshape((1, -1)), p2_event.reshape((1, -1)), memory_dist.reshape((1, -1)),
                                   indicator.reshape((1, -1))])
                if model_type == 'single':
                    inputs = torch.from_numpy(input).float().cuda()
                    mc, m21, m12, m1, m2, mg = net(inputs)
                    max_score, idx_mc = torch.max(mc, 1)
                    max_score, idx_m21 = torch.max(m21, 1)
                    max_score, idx_m12 = torch.max(m12, 1)
                    max_score, idx_m1 = torch.max(m1, 1)
                    max_score, idx_m2 = torch.max(m2, 1)
                    max_score, idx_mg = torch.max(mg, 1)

                    m1_predict.append(idx_m1.cpu().numpy()[0])
                    m2_predict.append(idx_m2.cpu().numpy()[0])
                    m12_predict.append(idx_m12.cpu().numpy()[0])
                    m21_predict.append(idx_m21.cpu().numpy()[0])
                    mc_predict.append(idx_mc.cpu().numpy()[0])
                    mg_predict.append(idx_mg.cpu().numpy()[0])
                else:
                    idx_m1 = clf1.predict(input)
                    idx_m2 = clf2.predict(input)
                    idx_m12 = clf12.predict(input)
                    idx_m21 = clf21.predict(input)
                    idx_mc = clfc.predict(input)
                    m1_predict.append(idx_m1[0])
                    m2_predict.append(idx_m2[0])
                    m12_predict.append(idx_m12[0])
                    m21_predict.append(idx_m21[0])
                    mc_predict.append(idx_mc[0])
                # update memory
                memory = update_memory(memory, 'm1', idx_m1, curr_loc)
                memory = update_memory(memory, 'm2', idx_m2, curr_loc)
                memory = update_memory(memory, 'm12', idx_m12, curr_loc)
                memory = update_memory(memory, 'm21', idx_m21, curr_loc)
                memory = update_memory(memory, 'mc', idx_mc, curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)
    print(metrics.classification_report(mc_real, mc_predict, digits=3))
    print(metrics.classification_report(m12_real, m12_predict, digits=3))
    print(metrics.classification_report(m21_real, m21_predict, digits=3))
    print(metrics.classification_report(m1_real, m1_predict, digits=3))
    print(metrics.classification_report(m2_real, m2_predict, digits=3))

def test_data_memory_gt_svm(prj_name):
    if prj_name == 'single':
        net = MLP()
        net.load_state_dict(torch.load('./cptk_single/model_350.pth'))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
    else:
        with open('./cptk_svm/clf_svm.p', 'rb') as f:
            clf1, clf2, clf12, clf21, clfc = joblib.load(f)

    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    obj_bbox_path = '/home/shuwen/data/data_preprocessing2/interpolate_bbox/'
    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    clips = os.listdir(event_label_path)
    m1_predict, m1_real = [], []
    m2_predict, m2_real = [], []
    m12_predict, m12_real = [], []
    m21_predict, m21_real = [], []
    mc_predict, mc_real = [], []
    mg_predict, mg_real = [], []
    for clip in clips[:10]:
        if not os.path.exists(reannotation_path + clip):
            continue
        # if not clip == 'test_94342_20.p':
        #     continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs, len(img_names))
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = annt.name.unique()
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        for obj_name in obj_names:
            if obj_name.startswith('P'):
                continue
            print(obj_name)
            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            for frame_id in range(p1_events_by_frame.shape[0]):
                # curr_loc
                curr_df = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                curr_loc = get_grid_location(curr_df)

                # gt
                # gt_obj_name, gt_bbox = get_obj_name(obj_bbox, annt, frame_id)
                # img = cv2.imread(img_names[frame_id])
                #
                # if not gt_obj_name:
                #     continue

                # cv2.rectangle(img, (int(obj_bbox[0]), int(obj_bbox[1])), (int(obj_bbox[2] + obj_bbox[0]), int(obj_bbox[3] + obj_bbox[1])), (255, 0, 0), thickness=2)
                # cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 0, 255), thickness=2)
                # cv2.imshow('img', img)
                # cv2.waitKey(200)
                obj_record = obj_records[obj_name][frame_id]

                for mind_name in obj_record:
                    if mind_name == 'm1':
                        m1_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm2':
                        m2_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm12':
                        m12_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm21':
                        m21_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mc':
                        mc_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mg':
                        mg_real.append(obj_record[mind_name]['fluent'])

                # event
                p1_event = p1_events_by_frame[frame_id]
                p2_event = p2_events_by_frame[frame_id]

                # memory
                memory_dist = []
                indicator = []
                for mind_name in memory.keys():
                    if frame_id == 0:
                        memory_dist.append(0)
                        indicator.append(0)
                    else:
                        if frame_id%50 == 0:
                            memory_loc = obj_records[obj_name][frame_id - 1][mind_name]['loc']
                        else:
                            memory_loc = memory[mind_name]['loc']
                        if memory_loc is not None:
                            curr_loc = np.array(curr_loc)
                            memory_loc = np.array(memory_loc)
                            memory_dist.append(np.linalg.norm(curr_loc - memory_loc))
                            indicator.append(1)
                        else:
                            memory_dist.append(0)
                            indicator.append(0)
                # get predicted value
                memory_dist = np.array(memory_dist)
                indicator = np.array(indicator)
                input = np.hstack([p1_event.reshape((1, -1)), p2_event.reshape((1, -1)), memory_dist.reshape((1, -1)),
                                   indicator.reshape((1, -1))])
                if prj_name == 'single':
                    inputs = torch.from_numpy(input).float().cuda()
                    m1, m2, m12, m21, mc = net(inputs)
                    max_score, idx_mc = torch.max(mc, 1)
                    max_score, idx_m21 = torch.max(m21, 1)
                    max_score, idx_m12 = torch.max(m12, 1)
                    max_score, idx_m1 = torch.max(m1, 1)
                    max_score, idx_m2 = torch.max(m2, 1)

                    m1_predict.append(idx_m1.cpu().numpy()[0])
                    m2_predict.append(idx_m2.cpu().numpy()[0])
                    m12_predict.append(idx_m12.cpu().numpy()[0])
                    m21_predict.append(idx_m21.cpu().numpy()[0])
                    mc_predict.append(idx_mc.cpu().numpy()[0])
                else:
                    idx_m1 = clf1.predict(input)
                    idx_m2 = clf2.predict(input)
                    idx_m12 = clf12.predict(input)
                    idx_m21 = clf21.predict(input)
                    idx_mc = clfc.predict(input)
                    m1_predict.append(idx_m1[0])
                    m2_predict.append(idx_m2[0])
                    m12_predict.append(idx_m12[0])
                    m21_predict.append(idx_m21[0])
                    mc_predict.append(idx_mc[0])

                # update memory
                memory = update_memory(memory, 'm1', idx_m1, curr_loc)
                memory = update_memory(memory, 'm2', idx_m2, curr_loc)
                memory = update_memory(memory, 'm12', idx_m12, curr_loc)
                memory = update_memory(memory, 'm21', idx_m21, curr_loc)
                memory = update_memory(memory, 'mc', idx_mc, curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)

    results_mc = metrics.classification_report(mc_real, mc_predict, digits=3)
    results_m1 = metrics.classification_report(m1_real, m1_predict, digits=3)
    results_m2 = metrics.classification_report(m2_real, m2_predict, digits=3)
    results_m12 = metrics.classification_report(m12_real, m12_predict, digits=3)
    results_m21 = metrics.classification_report(m21_real, m21_predict, digits=3)

    print(results_mc)
    print(results_m1)
    print(results_m2)
    print(results_m12)
    print(results_m21)

    with open('./cptk_' + prj_name + '/'  + 'seq.p', 'wb') as f:
        pickle.dump([results_m1, results_m2, results_m12, results_m21, results_mc], f)

def plot_confusion_matrix(cmc):
    df_cm = pd.DataFrame(cmc, range(cmc.shape[0]), range(cmc.shape[1]))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.show()

def test_data_memory_gt_lstm(prj_name):
    if prj_name == 'gru':
        net = MindGRU()
        net.load_state_dict(torch.load('./cptk_gru/model_350.pth'))
    else:
        net = MindLSTM()
        net.load_state_dict(torch.load('./cptk_lstm/model_350.pth'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    event_label_path = '/home/shuwen/data/data_preprocessing2/event_classfied_label/'
    reannotation_path = '/home/shuwen/data/data_preprocessing2/regenerate_annotation/'
    annotation_path = '/home/shuwen/data/data_preprocessing2/reformat_annotation/'
    color_img_path = '/home/shuwen/data/data_preprocessing2/annotations/'
    seq_len = 5
    with open('person_id.p', 'rb') as f:
        person_ids = pickle.load(f)
    clips = os.listdir(event_label_path)
    m1_predict, m1_real = [], []
    m2_predict, m2_real = [], []
    m12_predict, m12_real = [], []
    m21_predict, m21_real = [], []
    mc_predict, mc_real = [], []
    mg_predict, mg_real = [], []
    for clip in clips[:1]:
        if not os.path.exists(reannotation_path + clip):
            continue
        # if not clip == 'test_94342_20.p':
        #     continue
        print(clip)
        with open(reannotation_path + clip, 'rb') as f:
            obj_records = pickle.load(f)
        with open(event_label_path + clip, 'rb') as f:
            event_segs = pickle.load(f)
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        battery_events_by_frame, tracker_events_by_frame = reformat_events(event_segs, len(img_names))
        if person_ids[clip.split('.')[0]] == 'P1':
            p1_events_by_frame = tracker_events_by_frame
            p2_events_by_frame = battery_events_by_frame
        else:
            p1_events_by_frame = battery_events_by_frame
            p2_events_by_frame = tracker_events_by_frame
        annt = pd.read_csv(annotation_path + clip.split('.')[0] + '.txt', sep=",", header=None)
        annt.columns = ["obj_id", "x_min", "y_min", "x_max", "y_max", "frame", "lost", "occluded", "generated", "name",
                        "label"]
        obj_names = annt.name.unique()
        img_names = sorted(glob.glob(color_img_path + clip.split('.')[0] + '/kinect/*.jpg'))
        for obj_name in obj_names:
            if obj_name.startswith('P'):
                continue
            print(obj_name)
            memory = {'mg':{'fluent':None, 'loc':None}, 'mc':{'fluent':None, 'loc':None}, 'm21':{'fluent':None, 'loc':None},
                      'm12': {'fluent': None, 'loc': None}, 'm1':{'fluent':None, 'loc':None}, 'm2':{'fluent':None, 'loc':None}}
            for frame_id in range(p1_events_by_frame.shape[0]):
                event_input = np.zeros((seq_len, 10))
                memory_input = np.zeros((seq_len, 10))
                obj_record = obj_records[obj_name][frame_id]
                # flag = 0
                # for mind_name in obj_record:
                #     if not obj_record[mind_name]['fluent'] == 3:
                #         flag = 1
                # if flag == 0:
                #     continue
                for mind_name in obj_record:
                    if mind_name == 'm1':
                        m1_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm2':
                        m2_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm12':
                        m12_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'm21':
                        m21_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mc':
                        mc_real.append(obj_record[mind_name]['fluent'])
                    elif mind_name == 'mg':
                        mg_real.append(obj_record[mind_name]['fluent'])

                for i in range(-4, 1, 1):
                    curr_frame_id = max(frame_id + i, 0)
                    # curr_loc
                    curr_df = annt.loc[(annt.frame == curr_frame_id) & (annt.name == obj_name)]
                    curr_loc = get_grid_location(curr_df)
                    # event
                    p1_event = p1_events_by_frame[curr_frame_id]
                    p2_event = p2_events_by_frame[curr_frame_id]
                    event_input[i + 4, :5] = p1_event
                    event_input[i + 4, 5:] = p2_event
                    # memory
                    memory_dist = []
                    indicator = []
                    for mind_name in memory.keys():
                        if mind_name == 'mg':
                            continue
                        if curr_frame_id == 0:
                            memory_dist.append(0)
                            indicator.append(0)
                        else:
                            if frame_id%50 == 0:
                                memory_loc = obj_records[obj_name][curr_frame_id - 1][mind_name]['loc']
                            else:
                                memory_loc = memory[mind_name]['loc']
                            if memory_loc is not None:
                                curr_loc = np.array(curr_loc)
                                memory_loc = np.array(memory_loc)
                                memory_dist.append(np.linalg.norm(curr_loc - memory_loc))
                                indicator.append(1)
                            else:
                                memory_dist.append(0)
                                indicator.append(0)
                    # get predicted value
                    memory_dist = np.array(memory_dist)
                    indicator = np.array(indicator)
                    memory_input[i + 4, :5] = memory_dist
                    memory_input[i + 4, 5:] = indicator
                input = np.hstack([event_input, memory_input])
                input = input.reshape((1, input.shape[0], input.shape[1]))
                inputs = torch.from_numpy(input).float().cuda()
                m1, m2, m12, m21, mc = net(inputs)
                max_score, idx_mc = torch.max(mc, 1)
                max_score, idx_m21 = torch.max(m21, 1)
                max_score, idx_m12 = torch.max(m12, 1)
                max_score, idx_m1 = torch.max(m1, 1)
                max_score, idx_m2 = torch.max(m2, 1)
                m1_predict.append(idx_m1.cpu().numpy()[0])
                m2_predict.append(idx_m2.cpu().numpy()[0])
                m12_predict.append(idx_m12.cpu().numpy()[0])
                m21_predict.append(idx_m21.cpu().numpy()[0])
                mc_predict.append(idx_mc.cpu().numpy()[0])
                # update memory
                curr_df = annt.loc[(annt.frame == frame_id) & (annt.name == obj_name)]
                curr_loc = get_grid_location(curr_df)
                memory = update_memory(memory, 'm1', idx_m1, curr_loc)
                memory = update_memory(memory, 'm2', idx_m2, curr_loc)
                memory = update_memory(memory, 'm12', idx_m12, curr_loc)
                memory = update_memory(memory, 'm21', idx_m21, curr_loc)
                memory = update_memory(memory, 'mc', idx_mc, curr_loc)
                memory = update_memory(memory, 'mg', 2, curr_loc)

    results_mc = metrics.classification_report(mc_real, mc_predict, digits=3)
    results_m1 = metrics.classification_report(m1_real, m1_predict, digits=3)
    results_m2 = metrics.classification_report(m2_real, m2_predict, digits=3)
    results_m12 = metrics.classification_report(m12_real, m12_predict, digits=3)
    results_m21 = metrics.classification_report(m21_real, m21_predict, digits=3)

    print(results_mc)
    print(results_m1)
    print(results_m2)
    print(results_m12)
    print(results_m21)

    with open('./cptk_' + prj_name + '/' + 'seq.p', 'wb') as f:
        pickle.dump([results_m1, results_m2, results_m12, results_m21, results_mc], f)

    cmc = metrics.confusion_matrix(mc_real, mc_predict)
    cm1 = metrics.confusion_matrix(m1_real, m1_predict)
    cm2 = metrics.confusion_matrix(m2_real, m2_predict)
    cm12 = metrics.confusion_matrix(m12_real, m12_predict)
    cm21 = metrics.confusion_matrix(m21_real, m21_predict)

    plot_confusion_matrix(cmc)
    plot_confusion_matrix(cm1)
    plot_confusion_matrix(cm2)
    plot_confusion_matrix(cm12)
    plot_confusion_matrix(cm21)

# test_data_memory_gt_svm('svm')
# test_data_not_predict_svm()
# test_raw_data()
test_data_memory_gt_lstm('lstm')


# train('lstm_sep')
# net = MindLSTMSep()
# net.load_state_dict(torch.load('./cptk_lstm_sep/model_350.pth'))
# if torch.cuda.is_available():
#     net.cuda()
# net.eval()
# test_score(net, test_x, test_y, 2000, 'lstm_sep', 'test')

# test_svm(test_x, test_y, 'svm', 'test')
# train_svm()










