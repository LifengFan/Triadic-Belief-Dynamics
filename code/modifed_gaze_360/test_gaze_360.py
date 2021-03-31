import sys
import torch
import torchvision.transforms as transforms
import imageio
import cv2
import random
from PIL import Image
import math
import torch
import torchvision.transforms as transforms
import os
import glob
import numpy as np
import pandas as pd

sys.path.append('/home/shuwen/temp/gaze360/code/')
from model import GazeLSTM_modify
import joblib
import json
import moviepy.editor as mvp
from google.colab import files
import lucid.misc.io.showing as show
from lucid.misc.gl.glcontext import create_opengl_context
import OpenGL.GL as gl
from OpenGL.GL import shaders
WIDTH, HEIGHT = 960, 720
create_opengl_context((WIDTH, HEIGHT))

gl.glClear(gl.GL_COLOR_BUFFER_BIT)

image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

trackers = {'test1': 'skele1.p', 'test2': 'skele2.p', 'test6': 'skele2.p', 'test7': 'skele1.p',
                         'test_9434_1': 'skele2.p', 'test_9434_3': 'skele2.p', 'test_9434_18': 'skele1.p',
                         'test_94342_0': 'skele2.p', 'test_94342_1': 'skele2.p', 'test_94342_2': 'skele2.p',
                         'test_94342_3': 'skele2.p', 'test_94342_4': 'skele1.p', 'test_94342_5': 'skele1.p',
                         'test_94342_6': 'skele1.p', 'test_94342_7': 'skele1.p', 'test_94342_8': 'skele1.p',
                         'test_94342_10': 'skele2.p', 'test_94342_11': 'skele2.p', 'test_94342_12': 'skele1.p',
                         'test_94342_13': 'skele2.p', 'test_94342_14': 'skele1.p', 'test_94342_15': 'skele2.p',
                         'test_94342_16': 'skele1.p', 'test_94342_17': 'skele2.p', 'test_94342_18': 'skele1.p',
                         'test_94342_19': 'skele2.p', 'test_94342_20': 'skele1.p', 'test_94342_21': 'skele2.p',
                         'test_94342_22': 'skele1.p', 'test_94342_23': 'skele1.p', 'test_94342_24': 'skele1.p',
                         'test_94342_25': 'skele2.p', 'test_94342_26': 'skele1.p',
                         'test_boelter_1': 'skele2.p', 'test_boelter_2': 'skele2.p', 'test_boelter_3': 'skele2.p',
                         'test_boelter_4': 'skele1.p', 'test_boelter_5': 'skele1.p', 'test_boelter_6': 'skele1.p',
                         'test_boelter_7': 'skele1.p', 'test_boelter_9': 'skele1.p', 'test_boelter_10': 'skele1.p',
                         'test_boelter_12': 'skele2.p', 'test_boelter_13': 'skele1.p', 'test_boelter_14': 'skele1.p',
                         'test_boelter_15': 'skele1.p', 'test_boelter_17': 'skele2.p', 'test_boelter_18': 'skele1.p',
                         'test_boelter_19': 'skele2.p', 'test_boelter_21': 'skele1.p', 'test_boelter_22': 'skele2.p',
                         'test_boelter_24': 'skele1.p', 'test_boelter_25': 'skele1.p',
                         'test_boelter2_0': 'skele1.p', 'test_boelter2_2': 'skele1.p', 'test_boelter2_3': 'skele1.p',
                         'test_boelter2_4': 'skele1.p', 'test_boelter2_5': 'skele1.p', 'test_boelter2_6': 'skele1.p',
                         'test_boelter2_7': 'skele2.p', 'test_boelter2_8': 'skele2.p', 'test_boelter2_12': 'skele2.p',
                         'test_boelter2_14': 'skele2.p', 'test_boelter2_15': 'skele2.p', 'test_boelter2_16': 'skele1.p',
                         'test_boelter2_17': 'skele1.p',
                         'test_boelter3_0': 'skele1.p', 'test_boelter3_1': 'skele2.p', 'test_boelter3_2': 'skele2.p',
                         'test_boelter3_3': 'skele2.p', 'test_boelter3_4': 'skele1.p', 'test_boelter3_5': 'skele2.p',
                         'test_boelter3_6': 'skele2.p', 'test_boelter3_7': 'skele1.p', 'test_boelter3_8': 'skele2.p',
                         'test_boelter3_9': 'skele2.p', 'test_boelter3_10': 'skele1.p', 'test_boelter3_11': 'skele2.p',
                         'test_boelter3_12': 'skele2.p', 'test_boelter3_13': 'skele2.p',
                         'test_boelter4_0': 'skele2.p', 'test_boelter4_1': 'skele2.p', 'test_boelter4_2': 'skele2.p',
                         'test_boelter4_3': 'skele2.p', 'test_boelter4_4': 'skele2.p', 'test_boelter4_5': 'skele2.p',
                         'test_boelter4_6': 'skele2.p', 'test_boelter4_7': 'skele2.p', 'test_boelter4_8': 'skele2.p',
                         'test_boelter4_9': 'skele2.p', 'test_boelter4_10': 'skele2.p', 'test_boelter4_11': 'skele2.p',
                         'test_boelter4_12': 'skele2.p', 'test_boelter4_13': 'skele2.p',
                         }

frames = {'test_boelter2_16': [2, 14, 15, 16, 20, 22, 24, 25, 26, 27, 28, 35, 40, 45]}

def cal_mean_std():
    img_folder = './annotations/'
    clips = os.listdir(img_folder)
    count = 0
    mean_img = np.array([0.0, 0.0, 0.0])
    for clip in clips[:1]:
        print('*******'+clip+'*******')
        img_names = glob.glob(img_folder + clip + '/kinect/*.jpg')
        for img_name in img_names:
            img = cv2.imread(img_name)
            mean_img += np.array([img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()])
            count += 1
    mean_img = mean_img/count

    count = 0
    std_img = np.array([0, 0, 0])
    for clip in clips[:1]:
        print('*******' + clip + '*******')
        img_names = glob.glob(img_folder + clip + '/kinect/*.jpg')
        for img_name in img_names:
            img = cv2.imread(img_name)

            std_img += (np.array([img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()]) - mean_img)**2
            count += 1
    std_img = np.sqrt(std_img/count)

    return mean_img, std_img

def spherical2cartesial(x):
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output

def cal_angle(normal, obj):
    if np.linalg.norm(obj) > 0:
        obj = obj / np.linalg.norm(obj)
    if np.linalg.norm(normal) > 0:
        normal = normal / np.linalg.norm(normal)
    cosin = obj.dot(normal)
    return cosin


# def render_frame(x_position, y_position, vx, vy, vz, asize):
#     gl.glClear(gl.GL_COLOR_BUFFER_BIT)
#     with shader:
#         x_position = x_position * 0.89
#         y_position = y_position * 0.67
#         gl.glUniform1f(xpos, x_position)
#         gl.glUniform1f(ypos, y_position)
#
#         gl.glUniform1f(vdir_x, vx)
#         gl.glUniform1f(vdir_y, vy)
#         gl.glUniform1f(vdir_z, vz)
#         gl.glUniform1f(arrow_size, asize)
#
#         gl.glUniform3f(res_loc, WIDTH, HEIGHT, 1.0)
#
#         gl.glEnableVertexAttribArray(0);
#         gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, vertexPositions)
#         gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
#     img_buf = gl.glReadPixels(0, 0, WIDTH, HEIGHT, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
#     img = np.frombuffer(img_buf, np.uint8).reshape(HEIGHT, WIDTH, 3)[::-1]
#     return img

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


def extract_gaze(skeleton):
    skeleton = np.array(skeleton)
    a = skeleton[21] - skeleton[24]
    b = skeleton[22] - skeleton[24]
    normal = np.cross(a, b)
    if np.linalg.norm(normal) > 0:
        normal = normal / np.linalg.norm(normal)
    normal = normal + np.array([0, 0.8, 0])
    gaze_center = np.vstack([skeleton[21], skeleton[22], skeleton[24]]).mean(axis=0) + np.array([0, 0.2, 0])
    return normal, gaze_center

def record_skeleton():
    video_path = '../../data_preprocessing2/annotations/'
    save_path = './record_bbox/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    clips = os.listdir(video_path)
    for clip in clips:
        print('******************' + clip + '*****************')
        if os.path.exists(save_path + '/' + clip + '.p'):
            continue
        skeleton_path = '../../data_preprocessing2/post_skeletons/' + clip + '/'
        img_names = sorted(glob.glob('../../data_preprocessing2/annotations/' + clip + '/kinect/*.jpg'))

        if trackers[clip].split('.')[0][5] == '2':
            skele_file = skeleton_path + '/skele1.p'
        else:
            skele_file = skeleton_path + '/skele2.p'

        with open(skele_file, 'rb') as f:
            skeletons = joblib.load(f)

        # skele_file = trackers[clip]

        # with open(skeleton_path + '/' + skele_file, 'rb') as f:
        #     skeletons = joblib.load(f)


        to_draw_indices = [3, 4, 8, 21, 22, 23, 24, 25]
        tracking_id = dict()


        for i, skeleton_i in enumerate(skeletons):
            identity = dict()
            _, gaze_center = extract_gaze(skeleton_i)
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
            if np.mean(np.array(skeleton_i)) == 0:
                continue
            img = cv2.imread(img_names[i])
            to_draw_img = img.copy()
            to_draws = []
            for to_draw_index in to_draw_indices:
                depth = skeleton_i[to_draw_index][2]
                if depth > 4:
                    increment = np.array([0, 0.26, 0])
                else:
                    increment = np.array([0, 0.13, 0])
                to_draws.append(point2screen(skeleton_i[to_draw_index], increment))

            to_draws = np.array(to_draws)
            x_min = np.min(to_draws[:, 0])
            y_min = np.min(to_draws[:, 1])
            x_max = np.max(to_draws[:, 0])
            y_max = np.max(to_draws[:, 1])
            w = x_max - x_min
            h = y_max - y_min

            y_min = max(y_min - w*0.65, 0)
            y_max = min(y_max + w*0.65, to_draw_img.shape[1])
            x_min = max(x_min - h*0.55, 0)
            x_max = min(x_max + h*0.25, to_draw_img.shape[1])

            box = [x_min, y_min, x_max, y_max]
            # cv2.circle(to_draw_img, (int(gaze_screen[0]), int(gaze_screen[1])), 3, (255, 0, 0), thickness=1)
            # cv2.rectangle(to_draw_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
            # cv2.imshow('img', to_draw_img)
            # cv2.waitKey(20)

            identity[0] = [box, [gaze_screen[0], gaze_screen[1]], gaze_center]
            tracking_id[i] = identity

        with open(save_path + '/' + clip + '.p', 'wb') as f:
            joblib.dump(tracking_id, f)
        # return tracking_id

def record_skeleton_for_training():
    video_path = '../../data_preprocessing2/annotations/'
    gaze_path = '../../data_preprocessing2/gaze_training/'
    save_prefix = './gaze360_for_training/'

    clips = os.listdir(video_path)
    for clip in clips:
        print('******************' + clip + '*****************')
        skeleton_path = '../../data_preprocessing2/post_skeletons/' + clip + '/'
        img_names = sorted(glob.glob('../../data_preprocessing2/annotations/' + clip + '/kinect/*.jpg'))

        skele_file = skeleton_path + '/' + trackers[clip]

        with open(skele_file, 'rb') as f:
            skeletons = joblib.load(f)

        with open(gaze_path + clip + '/output.p', 'rb') as f:
            gazes = joblib.load(f)


        to_draw_indices = [3, 4, 8, 21, 22, 23, 24, 25]
        tracking_id = dict()

        save_path = save_prefix + clip + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, skeleton_i in enumerate(skeletons):
            if np.array(skeleton_i).mean() == 0:
                continue

            _, gaze_center = extract_gaze(skeleton_i)
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
            if np.mean(np.array(skeleton_i)) == 0:
                continue
            img = cv2.imread(img_names[i])
            to_draw_img = img.copy()
            to_draws = []
            for to_draw_index in to_draw_indices:
                depth = skeleton_i[to_draw_index][2]
                if depth > 4:
                    increment = np.array([0, 0.26, 0])
                else:
                    increment = np.array([0, 0.13, 0])
                to_draws.append(point2screen(skeleton_i[to_draw_index], increment))

            to_draws = np.array(to_draws)
            x_min = np.min(to_draws[:, 0])
            y_min = np.min(to_draws[:, 1])
            x_max = np.max(to_draws[:, 0])
            y_max = np.max(to_draws[:, 1])
            w = x_max - x_min
            h = y_max - y_min

            y_min = max(y_min - w*0.65, 0)
            y_max = min(y_max + w*0.65, to_draw_img.shape[1])
            x_min = max(x_min - h*0.55, 0)
            x_max = min(x_max + h*0.25, to_draw_img.shape[1])

            box = [x_min, y_min, x_max, y_max]
            cropped_image = to_draw_img[int(y_min):int(y_max), int(x_min):int(x_max), :]
            cv2.imwrite(save_path + '{0:04}'.format(i) + '.jpg', cropped_image)
            cv2.imshow('img', cropped_image)
            cv2.waitKey(20)




def gaze360_estimation():
    data_path = './annotations/'
    bbox_path = './record_bbox/'
    save_path = './record_gaze_normal_new_results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    clips = os.listdir(bbox_path)
    for clip in clips:
        clip = clip.split('.')[0]
        if os.path.exists(save_path + clip + '.p'):
            continue
        model = GazeLSTM_modify()
        model = torch.nn.DataParallel(model).cuda()
        model.cuda()
        # checkpoint = torch.load('/home/shuwen/temp/gaze360/gaze360_model.pth.tar')
        checkpoint = torch.load('/home/shuwen/temp/gaze360/code/model_best_Gaze360.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()


        img_names = sorted(glob.glob('../../data_preprocessing2/annotations/' + clip + '/kinect/*.jpg'))
        with open(bbox_path + clip + '.p', 'rb') as f:
            tracking_id = joblib.load(f)

        with open('/home/shuwen/data/data_preprocessing2/gaze360_test_skele_input_battery/' + clip + '.p', 'rb') as f:
            skeletons = joblib.load(f)

        gaze_dict = dict()
        for i in range(0, len(img_names)):
            print(i, len(img_names))
            image_ori = cv2.imread(img_names[i])

            if i in tracking_id and i in skeletons:
                for id_t in tracking_id[i].keys():
                    input_image = torch.zeros(7, 3, 224, 224)
                    input_skeletons = np.zeros((7, 26, 3))
                    count = 0
                    for j in range(i - 3, i + 4):
                        if j in tracking_id and id_t in tracking_id[j] and j>=0 and j<len(img_names):
                            new_im = Image.fromarray(cv2.imread(img_names[j]), 'RGB')
                            bbox, eyes, gaze_center = tracking_id[j][id_t]
                        else:
                            new_im = Image.fromarray(image_ori, 'RGB')
                            bbox, eyes, gaze_center = tracking_id[i][id_t]

                        if j in skeletons.keys():
                            input_skeletons[count, :, :] = np.array(skeletons[j])
                        else:
                            input_skeletons[count, :, :] = np.array(skeletons[i])

                        new_im = new_im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        input_image[count, :, :, :] = image_normalize(
                            transforms.ToTensor()(transforms.Resize((224, 224))(new_im)))
                        count = count + 1

                    input_skeletons[:, :, 0] = -1*input_skeletons[:, :, 0]
                    input_skeletons[:, :, 1] = -1 * input_skeletons[:, :, 1]
                    skeleton_float = torch.Tensor(input_skeletons)
                    skeleton_float = torch.FloatTensor(skeleton_float)
                    skeleton_float = skeleton_float.view(1, 7, 26 * 3)
                    output_gaze, _ = model(input_image.view(1, 7, 3, 224, 224).cuda(), skeleton_float.cuda())
                    gaze = spherical2cartesial(output_gaze).detach().numpy()
                    gaze = gaze.reshape((-1))
                    gaze_dict[i] = [gaze, gaze_center]

        with open(save_path + clip + '.p', 'wb') as f:
            joblib.dump(gaze_dict, f)

def gaze360_reformat_for_training():
    img_path = './gaze360_for_training/'
    save_path = './save_gaze360_label_reformat/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    clips = os.listdir(img_path)
    new_skeletons = []
    with open(save_path + 'gaze360_total_data_skele.txt', 'w+') as ft:
        for clip in clips:
            img_names= sorted(glob.glob(img_path + clip + '/*.jpg'))
            with open('../../data_preprocessing2/gaze360_label_reformat/' + clip + '.p', 'rb') as f:
                gaze_labels = joblib.load(f)
            f.close()

            with open('../../data_preprocessing2/gaze360_label_reformat_with_skele/' + clip + '.p', 'rb') as f:
                skeletons = joblib.load(f)
            f.close()


            for img_name_i in img_names:
                i = int(img_name_i.split('/')[-1].split('.')[0])
                if not i in gaze_labels.keys():
                    continue
                if not i in skeletons.keys():
                    continue

                img_name_prefix = '/'.join(img_name_i.split('/')[:-1])
                flag = 1
                for j in range(-3, 4):
                    if (not os.path.exists(img_name_prefix + '/' + '{0:04}'.format(i + j) + '.jpg')) or \
                        (not (i + j) in skeletons.keys()):
                        flag = 0
                        break


                if flag:
                    gaze = gaze_labels[i]
                    gaze = np.array(gaze)
                    if np.linalg.norm(gaze) > 0:
                        gaze = gaze/np.linalg.norm(gaze)
                    img_name_i_new = img_name_prefix[2:] + '/' + img_name_i.split('/')[-1]
                    ft.write(img_name_i_new + ' ' + str(-gaze[0]) + ' ' + str(-gaze[1]) + ' ' + str(gaze[2]) + '\n')
                    new_skeletons.append(skeletons[i])
    ft.close()

    # with open(save_path + 'skele.p', 'wb') as f:
    #     joblib.dump(new_skeletons, f)

    with open(save_path + 'gaze360_total_data_skele.txt', 'rb') as f:
        datas = f.readlines()


    index = list(range(len(datas)))
    random.shuffle(index)
    train_index = index[:len(index) // 2]
    validate_index = index[len(index) // 2:(len(index) // 2 + len(index) // 4)]
    test_index = index[(len(index) // 2 + len(index) // 4):]

    with open(save_path + 'gaze360_training_data_skele.txt', 'w+') as f:
        for i in train_index:
            f.write(datas[i])
    f.close()

    with open(save_path + 'gaze360_validate_data_skele.txt', 'w+') as f:
        for i in validate_index:
            f.write(datas[i])
    f.close()

    with open(save_path + 'gaze360_test_data_skele.txt', 'w+') as f:
        for i in test_index:
            f.write(datas[i])
    f.close()


def gaze360_visualization():
    data_path = './record_gaze_normal_training'
    clips = os.listdir(data_path)
    for clip in clips:
        clip = clip.split('.')[0]
        # if not clip == "test_boelter_9":
        #     continue
        # clip = 'test_94342_14'
        img_names = sorted(glob.glob('./annotations/' + clip + '/kinect/*.jpg'))
        with open('./transformed_normal_new_results/' + clip + '.p', 'rb') as f:
            gaze_normals = joblib.load(f)
        with open('./record_bbox/' + clip + '.p', 'rb') as f:
            tracking_id = joblib.load(f)
        # with open('../../data_preprocessing2/gaze_training/' + clip + '/others.p', 'rb') as f:
        #     gazes_gt = joblib.load(f)

        skeleton_path = './post_skeletons/' + clip + '/'

        # if trackers[clip].split('.')[0][5] == '2':
        #     skele_file = skeleton_path + '/skele1.p'
        # else:
        #     skele_file = skeleton_path + '/skele2.p'

        skele_file = skeleton_path + '/' + trackers[clip]
        with open(skele_file, 'rb') as f:
            skeletons = joblib.load(f)

        for i in range(0, len(img_names)):
            print(clip, i, len(img_names))
            image_ori = cv2.imread(img_names[i])
            image = image_ori.copy()
            image = image.astype(float)

            if i in gaze_normals:
                for id_t in tracking_id[i].keys():
                    bbox, eyes, gaze_center = tracking_id[i][id_t]
                    bbox = np.asarray(bbox).astype(int)

                    output_gaze = gaze_normals[i][0]
                    gaze_estimated_normal, gaze_estimated_center = extract_gaze(skeletons[i])
                    gaze_estimated_normal_center = point2screen(gaze_estimated_normal + gaze_center, np.array([0, 0, 0]))
                    # gaze_angle = cal_angle(gaze_estimated_normal, output_gaze - gaze_center)
                    # print(i, gaze_angle)
                    # # raw_input("Enter")
                    # if gaze_angle < 0.4:
                    #     output_gaze = gaze_estimated_normal + gaze_center
                    gaze_screen = point2screen(output_gaze, np.array([0, 0, 0]))
                    image = image.astype(np.uint8)
                    # if gazes_gt[i]:
                    #     gaze_gt = gazes_gt[i][1]
                    #     gaze_gt_screen = point2screen(gaze_gt, np.array([0, 0, 0]))
                    #     cv2.line(image, (int(gaze_gt_screen[0]), int(gaze_gt_screen[1])),
                    #              (int(eyes[0]), int(eyes[1])), (255, 0, 0), thickness=3)


                    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0))
                    cv2.line(image, (int(gaze_screen[0]), int(gaze_screen[1])),
                             (int(eyes[0]), int(eyes[1])), (255, 0, 255), thickness=3)
                    cv2.putText(image, 'frame:' + str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

            image = image.astype(np.uint8)
            cv2.imshow("img", image)
            cv2.waitKey(20)
            # raw_input("Enter")

if __name__ == '__main__':
    # calculating the mean and std of the input images
    mean, std = cal_mean_std()
    print(mean, std)

    # estimation head bouding box using head keypoints from skeletons
    record_skeleton()
    # estimation gaze direction from trained model
    gaze360_estimation()
    gaze360_visualization()

    # generate training data
    record_skeleton_for_training()
    gaze360_reformat_for_training()


