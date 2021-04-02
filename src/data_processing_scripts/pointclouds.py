import cv2
import numpy as np
import glob
import joblib
import os
import time
from joblib import Parallel, delayed
import open3d as o3d

mask_folder = '/home/yixin/Development/Data/masks/'
img_folder = '/home/yixin/Development/Data/kinect/'
save_folder = '/home/yixin/Development/Data/pointclouds/'
point_sparse = 20

K = [607.13232421875, 0.0, 638.6468505859375, 0.0, 607.1067504882812, 367.1607360839844, 0.0, 0.0, 1.0]
K = np.reshape(np.array(K), [3, 3])


def pixel2threed(x_d, y_d, depth, K):
    fx_d, fy_d, cx_d, cy_d = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if x_d < depth.shape[0] and y_d < depth.shape[1]:
        depth_value = depth[x_d, y_d]
        x = (y_d - cx_d) * depth_value / fx_d
        y = (x_d - cy_d) * depth_value / fy_d
        z = depth_value
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            x, y, z = 0, 0, 0
    else:
        x, y, z = 0, 0, 0
    return [x, y, z]


def convert2threed(input_path, save_path, idx, mask):
    if os.path.exists(save_path + '/' + '{0:04}'.format(idx) + '.p'):
        return
    depth_img = np.load(input_path)

    print("***img: " + str(idx))
    point3d = []
    for i_category in mask:
        category_points = []
        for i_object in i_category[1]:
            object_mask = i_object.todense()
            object_point_cloud = []
            if len(object_mask[object_mask == 1]) == 0:
                object_point_cloud.append([0, 0, 0])
                category_points.append(np.array(object_point_cloud))
                continue
            point_ids = list(zip(*np.where(object_mask == 1)))
            for i in range(0, len(point_ids)):
                point_id = point_ids[i]
                point = pixel2threed(point_id[0], point_id[1], depth_img, K)
                object_point_cloud.append(point)
            # down-sample object_point_cloud
            point_cloud_open3d = o3d.geometry.PointCloud()
            point_cloud_open3d.points = o3d.utility.Vector3dVector(object_point_cloud)
            point_cloud_open3d_down_sampled = point_cloud_open3d.voxel_down_sample(voxel_size=0.05)
            object_point_cloud_np = np.asarray(point_cloud_open3d_down_sampled.points)
            # save
            category_points.append(object_point_cloud_np)
        point3d.append([i_category[0], category_points])

    with open(save_path + '/' + '{0:04}'.format(idx) + '.p', 'wb') as f:
        joblib.dump(point3d, f)


def main():
    img_location_folder = os.listdir(img_folder)
    for i_folder in img_location_folder:
        all_clips = os.listdir(img_folder + '/' + i_folder)
        for i_clip in all_clips:
            print("start loading mask " + i_clip)
            mask_name = mask_folder + '/' + i_clip + '.p'
            if not os.path.exists(mask_name):
                continue
            with open(mask_name, 'rb') as f:
                masks = joblib.load(f)
            print("finish loading mask " + i_clip)

            depth_image_list = sorted(glob.glob(img_folder + i_folder + '/' + i_clip + '/depth*'))
            assert len(masks) == len(depth_image_list)
            save_path = save_folder + i_clip
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                continue

            print("start processing depth images in " + str(i_clip))
            start_time = time.time()
            Parallel(n_jobs=-1)(delayed(convert2threed)
                                (depth_image_list[idx], save_path, idx, masks[idx])
                                for idx in range(len(depth_image_list)))
            end_time = time.time()
            print("finish processing depth images in " + str(i_clip))
            print("time: ", end_time-start_time)


if __name__ == "__main__":
    main()
