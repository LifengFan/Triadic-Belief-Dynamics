import numpy as np
import torch
from torch.autograd import Variable
import pickle
from tqdm import tqdm
import glob
import cv2
import os
import metadata

with open("feature_ori.p", 'rb') as f:
    data = pickle.load(f)
print(data[0].shape)
with open("label.p", 'rb') as f:
    labels = pickle.load(f)

data = np.array(data)
labels = np.array(labels).reshape(-1)
ratio = len(data)*0.8
ratio = int(ratio)
indx = np.random.randint(0, len(data), ratio)
flags = np.zeros(len(data))
flags[indx] = 1
train_x, train_y = data[flags == 1], labels[flags == 1]
test_x, test_y = data[flags == 0], labels[flags == 0]

event_seg_tracker=metadata.event_seg_tracker


def test(net, data, label, batch_size):
    correct = 0
    total = 0
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

        predicted_val, embedding = net(torch.autograd.Variable(inputs))
        # convert 'predicted_val' GPU tensor to CPU tensor and extract the column with max_score
        predicted_val = predicted_val.data
        max_score, idx = torch.max(predicted_val, 1)
        assert idx.shape == actual_val.shape
        idx = idx.cpu().numpy()
        # compare it with actual value and estimate accuracy
        for i in range(idx.shape[0]):
            if idx[i] == actual_val[i]:
                correct += 1
        # pbar.set_description("processing batch %s" % str(batch_num))
    print("Classifier Accuracy: ", float(correct)/ data.shape[0])

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(381, 50)
        self.fc2 = torch.nn.Linear(50, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        embedding = self.fc1(x)
        out = self.relu(embedding)
        out = self.fc2(out)
        return out, embedding

learningRate = 0.01
epochs = 500
batch_size = 128

def train():
    model = MLP()
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
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
            inputs = torch.from_numpy(inputs_).float().cuda()
            labels = torch.from_numpy(labels_).long().reshape(-1).cuda()
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
            # Make gradients zero for parameters 'W', 'b'
            optimizer.zero_grad()
            forward_output, embedding = model(inputs)
            loss = criterion(forward_output, labels)
            loss.backward()
            optimizer.step()
            # calculating loss
            epoch_training_loss += loss.data.item()
            num_batches += 1
            # print(loss.data.item())
            # pbar.set_description("processing batch %s" % str(batch_num))

        print("epoch: ", epoch, ", loss: ", epoch_training_loss / num_batches)
        test(model, test_x, test_y, batch_size=2000)
        if epoch%10 == 0:
            save_path = './cptk/model_' +str(epoch)+'.pth'
            torch.save(model.state_dict(), save_path)

def find_parent_id(seg_id, seg_mark):
    while seg_mark[seg_id]:
        seg_id = seg_mark[seg_id]
    return seg_id

def concate_labels(labels):
    seg_mark = {}
    for seg_id, seg in enumerate(labels):
        if seg[1] - seg[0] < 10:
            if seg_id - 1 not in seg_mark:
                seg_mark[seg_id] = None
            if seg_id - 1 in seg_mark:
                seg_mark[seg_id] = seg_id - 1

    seg_id = len(labels) - 1
    to_remove = []
    while (seg_id > 0):
        if seg_id in seg_mark:
            parent_id = find_parent_id(seg_id, seg_mark)

            if parent_id != seg_id:
                labels[parent_id][1] = labels[seg_id][1]
                if labels[parent_id][1] - labels[parent_id][0] < 10:
                    labels[parent_id - 1][1] = labels[parent_id][1]
                    for i in np.arange(seg_id, parent_id - 1, -1):
                        to_remove.append(i)
                else:
                    for i in np.arange(seg_id, parent_id, -1):
                        to_remove.append(i)
            else:
                labels[seg_id - 1][1] = labels[seg_id][1]
                to_remove.append(seg_id)
            seg_id = parent_id - 1
        else:
            seg_id = seg_id - 1

    for i in range(len(to_remove)):
        del labels[to_remove[i]]

    ## added by Lifeng
    if (labels[0][1]-labels[0][0])<10:
        labels[1][0]=labels[0][0]
        del labels[0]

    return labels

def test_segments_pair(feature_file, img_files, clip):
    with open(feature_file, 'rb') as f:
        features = pickle.load(f)
    save_path = './seg_vis_pair/' #+ clip + '/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    net = MLP()
    net.load_state_dict(torch.load('./cptk/model_490.pth'))
    ##### For GPU #######
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    input = torch.from_numpy(features).float().cuda()
    predicted_val, embedding = net(torch.autograd.Variable(input))
    # convert 'predicted_val' GPU tensor to CPU tensor and extract the column with max_score
    predicted_val = predicted_val.data
    max_score, idx = torch.max(predicted_val, 1)

    # with open('embedding_feature.p', 'wb') as f:
    #     pickle.dump(feature_dict, f)
    idx = idx.cpu().numpy()

    labels = []
    labels.append([0])
    counter = 0
    for i in range(1, features.shape[0]):
        if idx[i] != idx[i - 1]:
            labels[counter].extend([i - 1, idx[i - 1]])
            labels.append([i])
            counter += 1
    if len(labels[counter]) < 2:
        labels[counter].extend([features.shape[0] - 1, idx[features.shape[0] - 1]])
    # labels = concate_labels(labels)
    # gt = event_seg_tracker[feature_file.split('/')[-1].split('.')[0]]
    # print(len(gt), len(labels))
    kmeans_seg = np.zeros(len(img_files))
    for seg_id, seg in enumerate(labels):
        if seg[1] == len(img_files) - 1:
            end = seg[1]
        else:
            end = seg[1] + 1
        kmeans_seg[seg[0]:end] = seg[2]
    # gt_seg = np.zeros(len(img_files))
    # for seg_id, seg in enumerate(gt):
    #     if seg[1] == len(img_files) - 1:
    #         end = seg[1]
    #     else:
    #         end = seg[1] + 1
    #     if seg[2] == 0:
    #         gt_seg[seg[0]:end] = 0
    #     else:
    #         gt_seg[seg[0]:end] = 1
    filename1 = save_path + clip + '.avi'
    video_shape = (800, 480)
    out = cv2.VideoWriter(filename1, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 24, video_shape)
    for frame_id, img_name in enumerate(img_files):
        # print(img_name)
        img = cv2.imread(img_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        # img = cv2.putText(img, 'gt:{}'.format(gt_seg[frame_id]), org, font,
        #                   fontScale, color, thickness, cv2.LINE_AA)
        img = cv2.putText(img, 'kmeans:{}'.format(kmeans_seg[frame_id]), (org[0], org[1] + 50), font,
                          fontScale, color, thickness, cv2.LINE_AA)
        # cv2.imwrite(save_path + '_{0:04}.jpg'.format(frame_id), img)
        out.write(cv2.resize(img, video_shape))
    out.release()

if __name__ == '__main__':
    # train()
    data_path = '/home/shuwen/data/data_preprocessing2/feature_pair/'
    img_path = '/home/shuwen/data/Six-Minds-Project/data_processing_scripts/seg_vis_pair/'
    clips = os.listdir(data_path)
    # clips = ['test_94342_16', 'test_boelter_12', 'test_boelter2_5', 'test_boelter3_4', 'test_boelter4_9']
    for clip in clips[:20]:
        clip = clip.split('.')[0]
        # if os.path.exists(img_path + clip):
        #     continue
        if clip in event_seg_tracker:
            continue
        if os.path.exists(img_path + clip + '.avi'):
            continue
        print(clip)
        img_names = sorted(glob.glob('/home/shuwen/data/data_preprocessing2/annotations/' + clip + '/kinect/*.jpg'))
        print(len(img_names))
        test_segments_pair(data_path + clip + '.p', img_names, clip)


    # for clip in clips:
    #     clip = clip.split('.')[0]
    #     os.system('ffmpeg -f image2 -i '+ img_path+clip+'/_%4d.jpg' + ' '+ img_path+clip+'_pair.mp4')