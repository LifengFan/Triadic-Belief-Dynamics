import numpy as np
import torch
from torch.autograd import Variable
import pickle
from tqdm import tqdm
import glob
import cv2
import os

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

event_seg_tracker = {'test_9434_18': [[0, 749, 0], [750, 824, 0], [825, 863, 2], [864, 974, 0], [975, 1041, 0]],
                             'test_94342_1': [[0, 13, 0], [14, 104, 0], [105, 333, 0], [334, 451, 0], [452, 652, 0],
                                              [653, 897, 0], [898, 1076, 0], [1077, 1181, 0], [1181, 1266, 0],
                                              [1267, 1386, 0]],
                             'test_94342_6': [[0, 95, 0], [96, 267, 1], [268, 441, 1], [442, 559, 1], [560, 681, 1], [
            682, 796, 1], [797, 835, 1], [836, 901, 0], [902, 943, 1]],
        'test_94342_10': [[0, 36, 0], [37, 169, 0], [170, 244, 1], [245, 424, 0], [425, 599, 0], [600, 640, 0],
                          [641, 680, 0], [681, 726, 1], [727, 866, 2], [867, 1155, 2]],
        'test_94342_21': [[0, 13, 0], [14, 66, 3], [67, 594, 2], [595, 1097, 2], [1098, 1133, 0]],
        'test1': [[0, 477, 0], [478, 559, 0], [560, 689, 2], [690, 698, 0]],
        'test6': [[0, 140, 0], [141, 375, 0], [375, 678, 0], [679, 703, 0]],
        'test7': [[0, 100, 0], [101, 220, 2], [221, 226, 0]],
        'test_boelter_2': [[0, 154, 0], [155, 279, 0], [280, 371, 0], [372, 450, 0], [451, 470, 0], [471, 531, 0],
                           [532, 606, 0]],
        'test_boelter_7': [[0, 69, 0], [70, 118, 1], [119, 239, 0], [240, 328, 1], [329, 376, 0], [377, 397, 1],
                           [398, 520, 0], [521, 564, 0], [565, 619, 1], [620, 688, 1], [689, 871, 0], [872, 897, 0],
                           [898, 958, 1], [959, 1010, 0], [1011, 1084, 0], [1085, 1140, 0], [1141, 1178, 0],
                           [1179, 1267, 1], [1268, 1317, 0], [1318, 1327, 0]],
        'test_boelter_24': [[0, 62, 0], [63, 185, 2], [186, 233, 2], [234, 292, 2], [293, 314, 0]],
        'test_boelter_12': [[0, 47, 1], [48, 119, 0], [120, 157, 1], [158, 231, 0], [232, 317, 0], [318, 423, 0], [424,
                                                                                                                  459,
                                                                                                                  0], [
                                           460, 522, 0], [523, 586, 0], [587, 636, 0], [637, 745, 1], [746, 971, 2]],
        'test_9434_1': [[0, 57, 0], [58, 124, 0], [125, 182, 1], [183, 251, 2],
                      [252, 417, 0]],
        'test_94342_16': [[0, 21, 0], [22, 45, 0], [46, 84, 0], [85, 158, 1], [159, 200, 1],
                        [201, 214, 0],
                        [215, 370, 1], [371, 524, 1], [525, 587, 3], [588, 782, 2],
                        [783, 1009, 2]],
        'test_boelter4_12': [[0, 141, 0], [142, 462, 2], [463, 605, 0], [606, 942, 2],
                           [943, 1232, 2], [1233, 1293, 0]],
        'test_boelter4_9': [[0, 27, 0], [28, 172, 0], [173, 221, 0], [222, 307, 1],
                          [308, 466, 0], [467, 794, 1], [795, 866, 1],
                          [867, 1005, 2], [1006, 1214, 2], [1215, 1270, 0]],
        'test_boelter4_4': [[0, 120, 0], [121, 183, 0], [184, 280, 1], [281, 714, 0]],
        'test_boelter4_3': [[0, 117, 0], [118, 200, 1], [201, 293, 1], [294, 404, 1],
                          [405, 600, 1], [601, 800, 1], [801, 905, 1],
                          [906, 1234, 1]],
        'test_boelter4_1': [[0, 310, 0], [311, 560, 0], [561, 680, 0], [681, 748, 0],
                          [749, 839, 0], [840, 1129, 0], [1130, 1237, 0]],
        'test_boelter3_13': [[0, 204, 2], [205, 300, 2], [301, 488, 2], [489, 755, 2]],
        'test_boelter3_11': [[0, 254, 1], [255, 424, 0], [425, 598, 1], [599, 692, 0],
                           [693, 772, 2], [773, 878, 2], [879, 960, 2], [961, 1171, 2],
                           [1172, 1397, 2]],
        'test_boelter3_6': [[0, 174, 1], [175, 280, 1], [281, 639, 0], [640, 695, 1],
                          [696, 788, 0], [789, 887, 2], [888, 1035, 1], [1036, 1445, 2]],
        'test_boelter3_4': [[0, 158, 1], [159, 309, 1], [310, 477, 1], [478, 668, 1],
                          [669, 780, 1], [781, 817, 0], [818, 848, 1], [849, 942, 1]],
        'test_boelter3_0': [[0, 140, 0], [141, 353, 0], [354, 599, 0], [600, 727, 0],
                          [728, 768, 0]],
        'test_boelter2_15': [[0, 46, 0], [47, 252, 2], [253, 298, 1], [299, 414, 2],
                           [415, 547, 2], [548, 690, 1], [691, 728, 1], [729, 773, 2],
                           [774, 935, 2]],
        'test_boelter2_12': [[0, 163, 0], [164, 285, 1], [286, 444, 1], [445, 519, 0],
                           [520, 583, 1], [584, 623, 0], [624, 660, 0],
                           [661, 854, 1], [855, 921, 1], [922, 1006, 2], [1007, 1125, 2],
                           [1126, 1332, 2], [1333, 1416, 2]],
        'test_boelter2_5': [[0, 94, 0], [95, 176, 1], [177, 246, 1], [247, 340, 1],
                          [341, 442, 1], [443, 547, 1], [548, 654, 1], [655, 734, 0],
                          [735, 792, 0], [793, 1019, 0], [1020, 1088, 0], [1089, 1206, 0],
                          [1207, 1316, 1], [1317, 1466, 1], [1467, 1787, 2],
                          [1788, 1936, 1], [1937, 2084, 2]],
        'test_boelter2_4': [[0, 260, 1], [261, 421, 1], [422, 635, 1], [636, 741, 1],
                          [742, 846, 1], [847, 903, 1], [904, 953, 1], [954, 1005, 1],
                          [1006, 1148, 1], [1149, 1270, 1], [1271, 1525, 1]],
        'test_boelter2_2': [[0, 131, 0], [132, 226, 0], [227, 267, 0], [268, 352, 0],
                          [353, 412, 0], [413, 457, 0], [458, 502, 0],
                          [503, 532, 0], [533, 578, 0], [579, 640, 0], [641, 722, 0],
                          [723, 826, 0], [827, 913, 0], [914, 992, 0],
                          [993, 1070, 0], [1071, 1265, 0], [1266, 1412, 0]],
        'test_boelter_21': [[0, 238, 1], [239, 310, 0], [311, 373, 1], [374, 457, 0],
                          [458, 546, 3], [547, 575, 1],
                          [576, 748, 2], [749, 952, 2]],
        }

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