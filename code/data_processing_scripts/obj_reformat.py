import os
import joblib

data_path = './post_images/kinect/'
video_folders = os.listdir(data_path)
for video_folder in video_folders:
    clips = os.listdir(data_path + video_folder)
    for clip in clips:
        # img_names = sorted(glob.glob('./post_images/kinect/' + video_folder + '/' + clip + '/*.jpg'))
        if not os.path.exists('./resort_detected_imgs/kinect/objs/' + clip + '.p'):
            continue

        save_path = './to_track/' + clip
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            continue
        print(clip)

        with open('./resort_detected_imgs/kinect/objs/' + clip + '.p', 'rb') as f:
            obj_frames = joblib.load(f)
        f.close()

        if not os.path.exists(save_path + '/det'):
            os.makedirs(save_path + '/det')
        with open(save_path + '/det/det.txt', 'w') as f:
            for frame_id, frame in enumerate(obj_frames):
                # img = cv2.imread(frame)
                # cv2.imwrite(save_path + '/img1/' + '{0:06}'.format(frame_id+1) + '.jpg', img)
                for objs in obj_frames[frame_id]:
                    for obj in objs[1]:
                        f.write(str(frame_id + 1) + ',' + str(-1) + ',' + str(obj[0]) + ',' + str(obj[1]) + ',' + str(
                            obj[2] - obj[0]) \
                                + ',' + str(obj[3] - obj[1]) + ',' + str(obj[4]) + ',' + str(-1) + ',' + str(
                            -1) + ',' + str(-1) + '\n')
        f.close()
        with open(save_path + '/seqinfo.ini', 'w') as f:
        
            f.write('[Sequence]\n')
            f.write('name=' + clip + '\n')
            f.write(
                'imDir=/home/shuwen/data/data_preprocessing2/post_images/kinect/' + video_folder + '/' + clip + '\n')
            f.write('frameRate=24\n')
            f.write('seqLength=' + str(len(obj_frames)) + '\n')
            f.write('imWidth=1280\n')
            f.write('imHeight=720\n')
            f.write('imExt=.jpg\n')
        f.close()
