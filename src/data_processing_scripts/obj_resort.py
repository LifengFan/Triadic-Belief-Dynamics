import glob
import pandas as pd
import joblib
import cv2
import os


class ObjResort:
    def __init__(self, obj_path='/home/shuwen/Downloads/detected_imgs/kinect/objs/',
                 mask_path='/home/shuwen/Downloads/detected_imgs/kinect/masks/',
                 rerecord_path='./before_resort/kinect/objs/'):
        self.obj_path = obj_path
        self.mask_path = mask_path
        self.rerecord_path = rerecord_path
        self.img_path = './post_images/kinect/'
        self.save_sorted_obj_path = './resort_detected_imgs/kinect/objs/'
        self.save_sorted_mask_path = './resort_detected_imgs/kinect/masks/'

    def rerecord(self):
        clips = sorted(glob.glob(self.obj_path + '*.p'))
        for clip in clips:
            clip_name = clip.split('/')[-1].split('.')[0]
            print(clip_name)
            save_path = self.rerecord_path + clip_name
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                continue
            with open(clip, 'rb') as f:
                frames = joblib.load(f)
            for frame_id, frame in enumerate(frames):
                with open(self.rerecord_path + clip_name + '/' + '{0:04}'.format(frame_id) + '.txt', 'w') as f:
                    for obj_id, obj in enumerate(frame):
                        for sub_id, sub_obj in enumerate(obj[1]):
                            if obj[0] == 'book':
                                ############check ratio################
                                # print((sub_obj[2] - sub_obj[0])/(sub_obj[3]-sub_obj[1]))
                                if (sub_obj[2] - sub_obj[0]) / (sub_obj[3] - sub_obj[1]) < 1:
                                    continue
                            f.write(obj[0] + ',' + str(obj_id) + ',' + str(sub_id) + ',' + str(sub_obj[0]) + ',' + \
                                    str(sub_obj[1]) + ',' + str(sub_obj[2]) + ',' + str(sub_obj[3]) + ',' + str(
                                sub_obj[4]) + '\n')
                f.close()

    def resort(self):
        video_folders = os.listdir(self.img_path)
        for video_folder in video_folders:
            clips = os.listdir(os.path.join(self.img_path, video_folder))
            for clip in clips:
                if os.path.exists(self.save_sorted_obj_path + clip + '.p'):
                    continue
                if not os.path.exists(os.path.join(self.rerecord_path, clip)):
                    continue
                print(clip)
                if not os.path.exists(self.save_sorted_obj_path):
                    os.makedirs(self.save_sorted_obj_path)

                if not os.path.exists(self.save_sorted_mask_path):
                    os.makedirs(self.save_sorted_mask_path)

                img_names = sorted(glob.glob(os.path.join(self.img_path, video_folder, clip) + '/*.jpg'))
                objs = []
                obj_masks_new = []
                with open(self.mask_path + clip + '.p', 'rb') as f:
                    obj_masks = joblib.load(f)
                for frame_id, img_name in enumerate(img_names):
                    objs.append([])
                    obj_masks_new.append([])
                    img = cv2.imread(img_name)
                    df = pd.read_csv(os.path.join(self.rerecord_path, clip) +
                                     '/' + '{0:04}'.format(frame_id) + '.txt', sep=",", header=None)
                    df.columns = ['category', 'obj_id', 'sub_id', 'left', 'top', 'right', 'bottom', 'conf']

                    if len(clip.split('_')) > 1 and clip.split('_')[1] == 'boelter4':
                        df_book = df.loc[df['category'] == 'book']
                        df_chair = df.loc[df['category'] == 'chair']
                        df_others = df.loc[(df['category'] != 'book') & (df['category'] != 'chair')]
                        df_others = df_others.sort_values(by=['conf'], ascending=False)
                        df_chair = df_chair.sort_values(by=['conf'], ascending=False)
                        obj_num = 25
                        if len(df_others) > obj_num:
                            df_others = df_others.iloc[:obj_num]
                        chair_num = 5
                        if len(df_chair) > chair_num:
                            df_chair = df_chair.iloc[:chair_num]
                        df_sort = pd.concat([df_book, df_chair, df_others])
                    else:
                        df_book = df.loc[df['category'] == 'book']
                        df_others = df.loc[df['category'] != 'book']
                        df_others = df_others.sort_values(by=['conf'], ascending=False)
                        obj_num = 25
                        if len(df_others) > obj_num:
                            df_others = df_others.iloc[:obj_num]
                        df_sort = pd.concat([df_book, df_others])
                    # for i, row in df_sort.iterrows():
                    #     cv2.rectangle(img, (int(row['left']), int(row['top'])), \
                    #         (int(row['right']), int(row['bottom'])), (255,0,255), thickness=3)
                    #     mask_temp = obj_masks[frame_id][row['obj_id']][1][row['sub_id']]
                    #     mask_temp = mask_temp.todense()
                    #     img[mask_temp] = 0.5*img[mask_temp] + [50, 0, 0]

                    # cv2.imshow(clip, img)
                    # cv2.waitKey(20)
                    # raw_input("Enter")
                    categories = df_sort['category'].unique()
                    boxes = []
                    masks = []
                    for cate_id, category in enumerate(categories):
                        boxes.append([])
                        masks.append([])
                        boxes[cate_id] = [category, []]
                        masks[cate_id] = [category, []]
                        sel_df = df_sort.loc[df_sort['category'] == category]
                        for i, row in sel_df.iterrows():
                            boxes[cate_id][1].append(
                                [row['left'], row['top'], row['right'], row['bottom'], row['conf']])
                            masks[cate_id][1].append(obj_masks[frame_id][row['obj_id']][1][row['sub_id']])
                    objs[frame_id] = boxes
                    obj_masks_new[frame_id] = masks
                with open(self.save_sorted_obj_path + clip + '.p', 'wb') as f:
                    joblib.dump(objs, f)
                with open(self.save_sorted_mask_path + clip + '.p', 'wb') as f:
                    joblib.dump(obj_masks_new, f)


if __name__ == "__main__":
    obj_resorter = ObjResort()
    # obj_resorter.rerecord()
    obj_resorter.resort()
