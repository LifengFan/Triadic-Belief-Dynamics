import cv2
import os
import glob

video_path = './raw_videos/eye_tracker_videos/'
frame_path = './raw_images/eye_tracker/'
video_folders = sorted(os.listdir(video_path))
for video_folder in video_folders:
    videos = sorted(glob.glob(video_path + video_folder + '/*.avi'))
    print(videos)
    for video in videos:
        video_name = video.split('/')[-1].split('.')[0]
        written_path = frame_path + video_folder + '/' + video_name + '/'
        print(written_path)
        if not os.path.exists(written_path):
            os.makedirs(written_path)
        print(video)
        cap = cv2.VideoCapture(video)
        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # cv2.imshow('frame', frame)
                # cv2.waitKey(20)
                cv2.imwrite(written_path + '{0:04}'.format(count) + '.jpg', frame)
                count += 1
                print(count)
            else:
                break
        cap.release()