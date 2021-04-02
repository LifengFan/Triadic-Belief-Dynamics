import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import glob
import numpy as np
from model import GazeLSTM_new
import joblib
from os import listdir

WIDTH, HEIGHT = 960, 720
image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def spherical2cartesial(x):
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output

def point2screen(points, increment):
    K = [607.13232421875, 0.0, 638.6468505859375, 0.0, 607.1067504882812, 367.1607360839844, 0.0, 0.0, 1.0]
    K = np.reshape(np.array(K), [3, 3])
    rot_points = np.array(points)
    rot_points = rot_points + increment
    points_camera = rot_points.reshape(3, 1)

    project_matrix = np.array(K).reshape(3, 3)
    points_prj = project_matrix.dot(points_camera)
    points_prj = points_prj.transpose()
    if not points_prj[:, 2][0] == 0.0:
        points_prj[:, 0] = points_prj[:, 0] / points_prj[:, 2]
        points_prj[:, 1] = points_prj[:, 1] / points_prj[:, 2]
    points_screen = points_prj[:, :2]
    assert points_screen.shape == (1, 2)
    points_screen = points_screen.reshape(-1)
    return points_screen

def get_bbx():

    skeleton_path = './skeleton/'
    img_names = sorted(glob.glob('./img/*.jpg'))

    skeleton_files=sorted(listdir(skeleton_path))
    N_p=len(skeleton_files)

    skeletons = {}
    for p_i in range(N_p):
        with open(skeleton_path+skeleton_files[p_i], 'rb') as f:
            skeletons[p_i]=joblib.load(f)

    to_draw_indices = [3, 4, 8, 21, 22, 23, 24, 25]

    for p_i in range(N_p):

        bbx= {}

        for i, skeleton in enumerate(skeletons[p_i]):
            if np.array(skeleton).mean() == 0:
                continue
            identity = dict()

            gaze_center = np.vstack([skeleton[21], skeleton[22], skeleton[24]]).mean(axis=0) + np.array([0, 0.2, 0])
            depth = gaze_center[2]
            if depth > 4:
                increment = np.array([0, 0.1, 0])
            elif depth > 3:
                increment = np.array([0, 0.05, 0])
            elif depth > 2:
                increment = np.array([0, -0.05, 0])
            else:
                increment = np.array([0, -0.1, 0])
            gaze_screen = point2screen(gaze_center, increment)
            # if np.mean(np.array(skeleton)) == 0:
            #     continue
            img = cv2.imread(img_names[i])
            to_draw_img = img.copy()
            to_draws = []
            for to_draw_index in to_draw_indices:
                depth = skeleton[to_draw_index][2]
                if depth > 4:
                    increment = np.array([0, 0.26, 0])
                else:
                    increment = np.array([0, 0.13, 0])
                to_draws.append(point2screen(skeleton[to_draw_index], increment))

            to_draws = np.array(to_draws)
            x_min = np.min(to_draws[:, 0])
            y_min = np.min(to_draws[:, 1])
            x_max = np.max(to_draws[:, 0])
            y_max = np.max(to_draws[:, 1])
            w = x_max - x_min
            h = y_max - y_min

            y_min = max(y_min - w * 0.65, 0)
            y_max = min(y_max + w * 0.65, to_draw_img.shape[1])
            x_min = max(x_min - h * 0.55, 0)
            x_max = min(x_max + h * 0.25, to_draw_img.shape[1])

            box = [x_min, y_min, x_max, y_max]

            identity[0] = [box, [gaze_screen[0], gaze_screen[1]], gaze_center]

            bbx[i]=identity

        with open('./bbx/bbx_'+str(p_i)+'.p', 'wb') as f:
            joblib.dump(bbx, f)

def gaze_prediction():

    bbox_path = './bbx/'
    skeleton_path = './skeleton/'
    save_path = './res/'

    model = GazeLSTM_new()
    model = torch.nn.DataParallel(model).cuda()
    model.cuda()

    checkpoint = torch.load('./model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    img_names = sorted(glob.glob('./img/*.jpg'))
    bbx_files=sorted(listdir(bbox_path))
    skeleton_files=sorted(listdir(skeleton_path))
    N_p=len(bbx_files)
    assert len(skeleton_files)==N_p

    # predict gaze
    for p_ind in range(N_p):

        with open('./bbx/bbx_'+str(p_ind)+'.p', 'rb') as f:
            bbx=joblib.load(f)

        with open(skeleton_path+skeleton_files[p_ind], 'rb') as f:
            skeletons=joblib.load(f)

        gaze_dict={}

        T=len(img_names)

        for f_i in range(T):
            image_ori = cv2.imread(img_names[f_i])

            #print p_ind, f_i, bbx[f_i].keys()

            if f_i in bbx.keys():
                for id_t in bbx[f_i].keys():
                    input_image = torch.zeros(7, 3, 224, 224)
                    input_skeletons = np.zeros((7, 26, 3))
                    count = 0
                    for j in range(f_i - 3, f_i + 4):
                        if j in bbx and id_t in bbx[j] and j >= 0 and j < len(img_names):
                            new_im = Image.fromarray(cv2.imread(img_names[j]), 'RGB')
                            bbox, eyes, gaze_center = bbx[j][id_t]
                        else:
                            new_im = Image.fromarray(image_ori, 'RGB')
                            bbox, eyes, gaze_center = bbx[f_i][id_t]

                        if j >= 0 and j < T:
                            input_skeletons[count, :, :] = np.array(skeletons[j])
                        else:
                            input_skeletons[count, :, :] = np.array(skeletons[f_i])

                        new_im = new_im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        input_image[count, :, :, :] = image_normalize(
                            transforms.ToTensor()(transforms.Resize((224, 224))(new_im)))
                        count = count + 1

                    input_skeletons[:, :, 0] = -1 * input_skeletons[:, :, 0]
                    input_skeletons[:, :, 1] = -1 * input_skeletons[:, :, 1]
                    skeleton_float = torch.Tensor(input_skeletons)
                    skeleton_float = torch.FloatTensor(skeleton_float)
                    skeleton_float = skeleton_float.view(1, 7, 26 * 3)
                    output_gaze, _ = model(input_image.view(1, 7, 3, 224, 224).cuda(), skeleton_float.cuda())
                    gaze = spherical2cartesial(output_gaze).detach().numpy()
                    gaze = gaze.reshape((-1))
                    gaze_dict[f_i] = [gaze, gaze_center]




        with open(save_path + 'res'+str(p_ind)+'.p', 'wb') as f:
                joblib.dump(gaze_dict, f)


def gaze_visualization():

    img_names = sorted(glob.glob('./img/*.jpg'))
    gaze_files=sorted(listdir('./res/'))
    N_p=len(gaze_files)

    gazes = {}
    bbxes = {}
    for p_i in range(N_p):
        with open('./res/res' + str(p_i) + '.p', 'rb') as f:
            gazes[p_i] = joblib.load(f)
        with open('./bbx/bbx_' + str(p_i) + '.p', 'rb') as f:
            bbxes[p_i] = joblib.load(f)


    for i in range(0, len(img_names)):
        image_ori = cv2.imread(img_names[i])
        image = image_ori.copy()
        image = image.astype(float)

        for p_i in range(N_p):

            if i in gazes[p_i]:
                for id_t in bbxes[p_i][i].keys():
                    bbox, eyes, gaze_center = bbxes[p_i][i][id_t]
                    bbox = np.asarray(bbox).astype(int)

                    output_gaze = gazes[p_i][i][0]
                    gaze_screen = point2screen(output_gaze, np.array([0, 0, 0]))
                    image = image.astype(np.uint8)

                    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0))
                    cv2.line(image, (int(gaze_screen[0]), int(gaze_screen[1])), (int(eyes[0]), int(eyes[1])),
                             (255, 0, 255), thickness=3)
                    cv2.putText(image, 'frame:' + str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

        image = image.astype(np.uint8)
        cv2.imshow("img", image)
        cv2.waitKey(20)


if __name__ == '__main__':

    # get_bbx()
    # gaze_prediction()
    gaze_visualization()
