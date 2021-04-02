import glob
import pickle
import numpy as np
import cv2
import os

frames = {"video_lib test1": [[142, 963], [0, 960], [740, 1439]],
          "video_lib test2": [[107, 1212], [231, 1483], [217, 1192]],
          "video_lib test6": [[129, 927], [64, 913], [168, 872]],
          "video_lib test7": [[181, 444], [79, 358], [49, 276]],
          "video_9434 test_9434_1": [[75, 533], [1212, 1740], [662, 1080]],
          "video_9434 test_9434_3": [[206, 611], [357, 879], [159, 482]],
          "video_9434 test_9434_18": [[15, 1186], [105, 1410], [44, 1086]],
          "video_94342 test_94342_0": [[2, 2110], [820, 3252], [595, 2535]],
          "video_94342 test_94342_1": [[62, 1455], [98, 1818], [90, 1477]],
          "video_94342 test_94342_2": [[37, 2139], [58, 2520], [81, 2049]],  # zhaoning stop at chocolate bottle
          "video_94342 test_94342_3": [[106, 2077], [84, 2328], [109, 1885]],  # jianxiong stops at dropping the bear
          "video_94342 test_94342_4": [[141, 2174], [108, 2515], [46, 1970]],  # jianxiong stops at touching the book
          "video_94342 test_94342_5": [[0, 2292], [210, 3054], [57, 2350]],  # jianxiong is looking at the banana
          "video_94342 test_94342_6": [[120, 1071], [94, 1237], [113, 1057]],  # jianxiong is holding the banana
          "video_94342 test_94342_7": [[44, 1474], [85, 1795], [29, 1415]],
          "video_94342 test_94342_8": [[52, 1930], [111, 2278], [79, 1874]],
          "video_94342 test_94342_10": [[40, 1282], [88, 1534], [39, 1195]],
          "video_94342 test_94342_11": [[44, 1775], [125, 2132], [28, 1638]],  # jianxiong thmubs up
          "video_94342 test_94342_12": [[66, 2549], [103, 3110], [94, 2511]],
          "video_94342 test_94342_13": [[62, 1524], [50, 1860], [67, 1522]],  # jianxiong is watching the clock
          "video_94342 test_94342_14": [[72, 1951], [84, 2392], [85, 1926]],  # zhaoning is touching the bag
          "video_94342 test_94342_15": [[12, 1249], [64, 1530], [103, 1277]],  # zhaoning goes towards the bear
          "video_94342 test_94342_16": [[296, 1317], [62, 1317], [67, 1077]],  # jianxiong is watching the ball
          "video_94342 test_94342_17": [[298, 1355], [338, 1634], [156, 1214]],
          "video_94342 test_94342_18": [[81, 1661], [80, 1986], [69, 1608]],
          "video_94342 test_94342_19": [[79, 1822], [69, 2243], [93, 1788]],  # zhaoning is playting the ball
          "video_94342 test_94342_20": [[29, 1906], [65, 2354], [91, 1956]],
          "video_94342 test_94342_21": [[61, 1250], [82, 1533], [55, 1189]],  # jianxiong is holding the bear
          "video_94342 test_94342_22": [[54, 647], [50, 768], [41, 627]],
          "video_94342 test_94342_23": [[102, 651], [71, 725], [78, 617]],
          "video_94342 test_94342_24": [[25, 766], [48, 973], [55, 806]],
          "video_94342 test_94342_25": [[3, 623], [90, 850], [26, 594]],
          "video_94342 test_94342_26": [[91, 627], [89, 779], [66, 593]],
          "video_bhlib test_boelter_1": [[435, 1442], [247, 1391], [522, 1447]],
          # xiaojian looks books, sirui look into bag
          "video_bhlib test_boelter_2": [[215, 866], [419, 1182], [800, 1407]],
          "video_bhlib test_boelter_3": [[54, 650], [181, 926], [212, 821]],
          "video_bhlib test_boelter_4": [[297, 1085], [109, 1077], [211, 998]],
          "video_bhlib test_boelter_5": [[55, 889], [72, 1110], [102, 936]],
          # sirui drops the pink bup, xiaojian looks at clock
          "video_bhlib test_boelter_6": [[72, 1616], [160, 2036], [174, 1609]],
          # xiaojian drops the banana, sisui looks at book
          "video_bhlib test_boelter_7": [[55, 1502], [105, 1768], [357, 1685]],
          # xiaojian drops the clock, sirui looks at green book
          "video_bhlib test_boelter_9": [[126, 814], [167, 968], [140, 776]],
          "video_bhlib test_boelter_10": [[76, 775], [120, 934], [128, 782]],
          "video_bhlib test_boelter_12": [[63, 1150], [194, 1428], [209, 1181]],
          "video_bhlib test_boelter_13": [[88, 1261], [482, 1766], [245, 1249]],
          "video_bhlib test_boelter_14": [[78, 1046], [152, 1348], [203, 1107]],
          "video_bhlib test_boelter_15": [[96, 1230], [391, 1744], [507, 1562]],
          # xiaojian saw the clock, sirui points to himself
          "video_bhlib test_boelter_17": [[101, 943], [112, 1082], [182, 999]],
          "video_bhlib test_boelter_18": [[38, 1424], [194, 1942], [186, 1575]],
          "video_bhlib test_boelter_19": [[156, 1115], [130, 1383], [255, 1238]],
          "video_bhlib test_boelter_21": [[184, 1137], [175, 1419], [163, 1127]],
          "video_bhlib test_boelter_22": [[126, 980], [533, 1605], [496, 1324]],
          "video_bhlib test_boelter_24": [[73, 388], [468, 891], [203, 527]],
          "video_bhlib test_boelter_25": [[507, 1458], [241, 1473], [551, 1539]],
          "video_bhlib2 test_boelter2_0": [[50, 1308], [236, 1683], [193, 1336]],
          # yangyu gets the book, wangshu put the apple
          "video_bhlib2 test_boelter2_2": [[201, 1718], [153, 1905], [171, 1584]],  # both are looking at books
          "video_bhlib2 test_boelter2_3": [[23, 970], [112, 1256], [692, 1634]],
          "video_bhlib2 test_boelter2_4": [[75, 1621], [234, 2156], [245, 1771]],  # wangshu gets the chocolate bottle
          "video_bhlib2 test_boelter2_5": [[56, 2255], [75, 2682], [100, 2185]],
          # yangyu stands up, wangshu leans forward
          "video_bhlib2 test_boelter2_6": [[86, 2265], [105, 2680], [76, 2176]],
          # wangshu looks into case, yangyu looks at case
          "video_bhlib2 test_boelter2_7": [[312, 1894], [111, 1993], [315, 1820]],  # waangshu sleeps
          "video_bhlib2 test_boelter2_8": [[88, 1466], [170, 1841], [66, 1413]],
          # wangshu picks the bear, yangyu looks the bear
          "video_bhlib2 test_boelter2_12": [[22, 1470], [128, 1891], [109, 1526]],
          "video_bhlib2 test_boelter2_14": [[61, 981], [382, 1499], [128, 1070]],
          "video_bhlib2 test_boelter2_15": [[76, 1039], [115, 1285], [256, 1192]],
          # wangshu picks the book, yangyu after pointing
          "video_bhlib2 test_boelter2_16": [[186, 1920], [343, 2515], [249, 1987]],  # cheers up
          "video_bhlib2 test_boelter2_17": [[62, 1369], [167, 1741], [149, 1417]],
          "video_bhlib3 test_boelter3_0": [[0, 783], [400, 1332], [331, 1100]],
          "video_bhlib3 test_boelter3_1": [[0, 1510], [240, 1907], [209, 1560]],
          # lupan looks the side of the bear, shuwen looks green bottle
          "video_bhlib3 test_boelter3_2": [[128, 1518], [69, 1697], [73, 1399]],
          # lupan pick the cup, shuwen looks at bottle
          "video_bhlib3 test_boelter3_3": [[192, 1063], [147, 1200], [120, 981]],
          "video_bhlib3 test_boelter3_4": [[95, 1038], [133, 1353], [145, 1112]],
          "video_bhlib3 test_boelter3_5": [[153, 1085], [152, 1260], [145, 996]],
          "video_bhlib3 test_boelter3_6": [[397, 1902], [75, 1865], [330, 1776]],
          "video_bhlib3 test_boelter3_7": [[144, 1798], [90, 2021], [115, 1659]],
          "video_bhlib3 test_boelter3_8": [[120, 1336], [235, 1703], [236, 1397]],
          # lupan put the cup, shuwen sees the cup
          "video_bhlib3 test_boelter3_9": [[158, 1589], [106, 1854], [140, 1523]],  # lupan sees shuwen see the book
          "video_bhlib3 test_boelter3_10": [[173, 1474], [153, 1772], [117, 1426]],
          # shuwen lift the bear lupan sees the bear
          "video_bhlib3 test_boelter3_11": [[79, 1477], [127, 1893], [166, 1605]],
          "video_bhlib3 test_boelter3_12": [[188, 547], [99, 619], [106, 517]],
          "video_bhlib3 test_boelter3_13": [[172, 928], [36, 1102], [125, 999]],
          "video_bhlib4 test_boelter4_0": [[84, 1487], [383, 2056], [182, 1504]],
          "video_bhlib4 test_boelter4_1": [[122, 1461], [139, 1705], [89, 1327]],
          "video_bhlib4 test_boelter4_2": [[220, 1635], [85, 1771], [234, 1587]],
          "video_bhlib4 test_boelter4_3": [[99, 1406], [214, 1751], [94, 1329]],  # yining saw shuwen put the green book
          "video_bhlib4 test_boelter4_4": [[71, 902], [350, 1212], [143, 858]],
          # shuwen see the book, yining see the door
          "video_bhlib4 test_boelter4_5": [[40, 1259], [111, 1590], [238, 1404]],
          "video_bhlib4 test_boelter4_6": [[75, 1440], [150, 1820], [94, 1428]],
          "video_bhlib4 test_boelter4_7": [[80, 1505], [267, 1882], [193, 1594]],
          "video_bhlib4 test_boelter4_8": [[74, 1100], [148, 1400], [83, 1089]],
          "video_bhlib4 test_boelter4_9": [[0, 1307], [125, 1719], [85, 1356]],
          # shuwen put the banana, yining see the laptop
          "video_bhlib4 test_boelter4_10": [[104, 1419], [169, 1745], [102, 1365]],
          "video_bhlib4 test_boelter4_11": [[88, 1496], [162, 1868], [587, 1942]],
          # shuwen stands up, yining see the laptop
          "video_bhlib4 test_boelter4_12": [[115, 1486], [248, 1889], [42, 1336]],
          # yining put the pink cup, shuwen see yining
          "video_bhlib4 test_boelter4_13": [[386, 1413], [432, 1617], [59, 992]]
          }

save_add_path = True
data_path = './raw_images/kinect/'
video_folders = os.listdir(data_path)
for video_folder in video_folders:
    clips = os.listdir(data_path + video_folder)
    for clip in clips:
        # for key, frame in frames.items():
        key = video_folder + ' ' + clip
        frame = frames[key]
        print(key)
        clip_name = key.split()[1]

        path_name = os.path.join(key.split()[0], key.split()[1])
        gaze_post_path = './post_images/tracker/' + path_name + '/' + clip_name
        kinect_color_frames = sorted(glob.glob('raw_images/kinect/' + path_name + '/color*'))
        kinect_depth_frames = sorted(glob.glob('raw_images/kinect/' + path_name + '/depth*'))
        with open('raw_images/kinect/' + path_name + '/skele1.p') as f:
            skele1 = pickle.load(f)
        with open('raw_images/kinect/' + path_name + '/skele2.p') as f:
            skele2 = pickle.load(f)

        battery_color_frames = sorted(glob.glob('../data_preprocessing/raw_images/battery/' + path_name + '/*.jpg'))

        eye_tracker_color_frames = sorted(
            glob.glob('../data_preprocessing/raw_images/eye_tracker/' + path_name + '/*.jpg'))
        with open('../data_preprocessing/gaze/' + path_name + '.p', 'rb') as f:
            gazes = pickle.load(f)
            print(len(gazes))

        kinect_start_frame = frame[0][0]
        kinect_end_frame = frame[0][1]
        kinect_count = kinect_end_frame - kinect_start_frame

        battery_start_frame = frame[1][0]
        battery_end_frame = frame[1][1]
        battery_count = battery_end_frame - battery_start_frame

        tracker_start_frame = frame[2][0]
        tracker_end_frame = frame[2][1]
        tracker_count = tracker_end_frame - tracker_start_frame

        min_id = np.argmin(np.array([kinect_count, battery_count, tracker_count]))
        min_value = min(kinect_count, battery_count, tracker_count)

        kinect_post_path = './post_images/kinect/' + path_name + '/'
        battery_post_path = './post_images/battery/' + path_name + '/'
        tracker_post_path = './post_images/tracker/' + path_name + '/'

        if not os.path.exists(kinect_post_path):
            os.makedirs(kinect_post_path)
        else:
            continue
            # if path_name is not 'video_lib' and path_name is not 'video_9434':
            #     continue
        if not os.path.exists(battery_post_path):
            os.makedirs(battery_post_path)
        if not os.path.exists(tracker_post_path):
            os.makedirs(tracker_post_path)

        if min_id == 1:
            print("dumping kinect")
            count = 0
            new_skele1 = []
            new_skele2 = []
            frame_idx = np.arange(0, min_value)
            ratio = kinect_count / float(min_value)
            frame_idx = frame_idx * ratio
            frame_idx = np.round(frame_idx) + kinect_start_frame

            assert frame_idx.shape[0] == min_value

            if save_add_path:
                kinect_post_path2 = './annotations/' + clip_name + '/kinect/'
                if not os.path.exists(kinect_post_path2):
                    os.makedirs(kinect_post_path2)
            for i in frame_idx:
                i = int(i)
                color_img = cv2.imread(kinect_color_frames[i])
                depth_img = np.load(kinect_depth_frames[i])
                # depth_img = cv2.imread(kinect_depth_frames[i])
                img_name = kinect_color_frames[i].split('/')[-1].split('.')[0]
                cv2.imwrite(kinect_post_path + 'color_' + '{0:04}'.format(count) + '_' + img_name + '.jpg', color_img)
                np.save(kinect_post_path + 'depth_' + '{0:04}'.format(count) + '_' + img_name + '.npy', depth_img)
                # cv2.imwrite(kinect_post_path + 'depth_' + '{0:04}'.format(count) + '_' + img_name + '.jpg', depth_img)
                new_skele1.append(skele1[i])
                new_skele2.append(skele2[i])
                if save_add_path:
                    cv2.imwrite(kinect_post_path2 + '{0:04}'.format(count) + '_' + img_name + '.jpg', color_img)

                count += 1

            with open(kinect_post_path + 'skele1.p', 'wb') as f:
                pickle.dump(new_skele1, f)
            with open(kinect_post_path + 'skele2.p', 'wb') as f:
                pickle.dump(new_skele2, f)

            print("dumping tracker")
            count = 0
            frame_idx = np.arange(0, min_value)
            ratio = tracker_count / float(min_value)
            frame_idx = frame_idx * ratio
            frame_idx = np.round(frame_idx) + tracker_start_frame
            assert frame_idx.shape[0] == min_value

            gazes_new = []
            count = 0
            if save_add_path:
                tracker_post_path2 = './annotations/' + clip_name + '/tracker/'
                if not os.path.exists(tracker_post_path2):
                    os.makedirs(tracker_post_path2)
            for i in frame_idx:
                i = int(i)
                img = cv2.imread(eye_tracker_color_frames[i])
                img_name = eye_tracker_color_frames[i].split('/')[-1].split('.')[0]
                gaze = gazes[i]
                if not np.isnan(gaze['gaze'][0]) and not np.isnan(gaze['gaze'][1]):
                    cv2.circle(img, (int(gaze['gaze'][0]), int(gaze['gaze'][1])), 7, (255, 0, 255), thickness=5)
                cv2.imwrite(tracker_post_path + '{0:04}'.format(count) + '_' + img_name + '.jpg', img)
                gazes_new.append(gazes[i])
                if save_add_path:
                    cv2.imwrite(tracker_post_path2 + '{0:04}'.format(count) + '_' + img_name + '.jpg', img)
                count += 1

            with open(gaze_post_path + '.p', 'wb') as f:
                pickle.dump(gazes_new, f)

            print("dumping battery")
            count = 0
            if save_add_path:
                battery_post_path2 = './annotations/' + clip_name + '/battery/'
                if not os.path.exists(battery_post_path2):
                    os.makedirs(battery_post_path2)
            for i in range(battery_start_frame, battery_end_frame):
                img_name = battery_color_frames[i].split('/')[-1].split('.')[0]
                img = cv2.imread(battery_color_frames[i])
                cv2.imwrite(battery_post_path + img_name + '.jpg', img)

                if save_add_path:
                    cv2.imwrite(battery_post_path2 + '{0:04}'.format(count) + '_' + img_name + '.jpg', img)
                count += 1


        elif min_id == 0:
            print("dumping battery")
            count = 0
            frame_idx = np.arange(0, min_value)
            ratio = battery_count / float(min_value)
            frame_idx = frame_idx * ratio
            frame_idx = np.round(frame_idx) + battery_start_frame
            assert frame_idx.shape[0] == min_value

            if save_add_path:
                battery_post_path2 = './annotations/' + clip_name + '/battery/'
                if not os.path.exists(battery_post_path2):
                    os.makedirs(battery_post_path2)
            for i in frame_idx:
                i = int(i)
                print(i, len(battery_color_frames))
                color_img = cv2.imread(battery_color_frames[i])
                img_name = battery_color_frames[i].split('/')[-1].split('.')[0]
                cv2.imwrite(battery_post_path + '{0:04}'.format(count) + '_' + img_name + '.jpg', color_img)

                if save_add_path:
                    cv2.imwrite(battery_post_path2 + '{0:04}'.format(count) + '_' + img_name + '.jpg', color_img)
                count += 1

            print("dumping tracker")
            count = 0
            frame_idx = np.arange(0, min_value)
            ratio = tracker_count / float(min_value)
            frame_idx = frame_idx * ratio
            frame_idx = np.round(frame_idx) + tracker_start_frame
            assert frame_idx.shape[0] == min_value

            gazes_new = []
            if save_add_path:
                tracker_post_path2 = './annotations/' + clip_name + '/tracker/'
                if not os.path.exists(tracker_post_path2):
                    os.makedirs(tracker_post_path2)
            for i in frame_idx:
                i = int(i)
                img = cv2.imread(eye_tracker_color_frames[i])
                img_name = eye_tracker_color_frames[i].split('/')[-1].split('.')[0]
                gaze = gazes[i]
                if not np.isnan(gaze['gaze'][0]) and not np.isnan(gaze['gaze'][1]):
                    cv2.circle(img, (int(gaze['gaze'][0]), int(gaze['gaze'][1])), 7, (255, 0, 255), thickness=5)
                cv2.imwrite(tracker_post_path + '{0:04}'.format(count) + '_' + img_name + '.jpg', img)
                gazes_new.append(gazes[i])
                if save_add_path:
                    cv2.imwrite(tracker_post_path2 + '{0:04}'.format(count) + '_' + img_name + '.jpg', img)
                count += 1

            with open(gaze_post_path + '.p', 'wb') as f:
                pickle.dump(gazes_new, f)

            print('dumping kinect')
            count = 0
            if save_add_path:
                kinect_post_path2 = './annotations/' + clip_name + '/kinect/'
                if not os.path.exists(kinect_post_path2):
                    os.makedirs(kinect_post_path2)
            for i in range(kinect_start_frame, kinect_end_frame):
                color_img = cv2.imread(kinect_color_frames[i])
                depth_img = np.load(kinect_depth_frames[i])
                # depth_img = cv2.imread(kinect_depth_frames[i])
                img_name = kinect_color_frames[i].split('/')[-1].split('.')[0]
                cv2.imwrite(kinect_post_path + 'color_' + '{0:04}'.format(count) + '_' + img_name + '.jpg', color_img)
                np.save(kinect_post_path + 'depth_' + '{0:04}'.format(count) + '_' + img_name + '.npy', depth_img)
                if save_add_path:
                    cv2.imwrite(kinect_post_path2 + '{0:04}'.format(count) + '_' + img_name + '.jpg', color_img)
                count += 1
                # cv2.imwrite(kinect_post_path + 'depth_' + img_name + '.jpg', depth_img)

            with open(kinect_post_path + 'skele1.p', 'wb') as f:
                pickle.dump(skele1[kinect_start_frame:kinect_end_frame], f)
            with open(kinect_post_path + 'skele2.p', 'wb') as f:
                pickle.dump(skele2[kinect_start_frame:kinect_end_frame], f)

        elif min_id == 2:
            print("dumping battery")
            count = 0
            frame_idx = np.arange(0, min_value)
            ratio = battery_count / float(min_value)
            frame_idx = frame_idx * ratio
            frame_idx = np.round(frame_idx) + battery_start_frame
            assert frame_idx.shape[0] == min_value

            if save_add_path:
                battery_post_path2 = './annotations/' + clip_name + '/battery/'
                if not os.path.exists(battery_post_path2):
                    os.makedirs(battery_post_path2)
            for i in frame_idx:
                i = int(i)
                color_img = cv2.imread(battery_color_frames[i])
                img_name = battery_color_frames[i].split('/')[-1].split('.')[0]
                cv2.imwrite(battery_post_path + '{0:04}'.format(count) + '_' + img_name + '.jpg', color_img)
                if save_add_path:
                    cv2.imwrite(battery_post_path2 + '{0:04}'.format(count) + '_' + img_name + '.jpg', color_img)
                count += 1

            print("dumping kinect")
            count = 0
            new_skele1 = []
            new_skele2 = []
            frame_idx = np.arange(0, min_value)
            ratio = kinect_count / float(min_value)
            frame_idx = frame_idx * ratio
            frame_idx = np.round(frame_idx) + kinect_start_frame

            assert frame_idx.shape[0] == min_value

            if save_add_path:
                kinect_post_path2 = './annotations/' + clip_name + '/kinect/'
                if not os.path.exists(kinect_post_path2):
                    os.makedirs(kinect_post_path2)
            for i in frame_idx:
                i = int(i)
                color_img = cv2.imread(kinect_color_frames[i])
                depth_img = np.load(kinect_depth_frames[i])
                # depth_img = cv2.imread(kinect_depth_frames[i])
                img_name = kinect_color_frames[i].split('/')[-1].split('.')[0]
                cv2.imwrite(kinect_post_path + 'color_' + '{0:04}'.format(count) + '_' + img_name + '.jpg', color_img)
                np.save(kinect_post_path + 'depth_' + '{0:04}'.format(count) + '_' + img_name + '.npy', depth_img)
                # cv2.imwrite(kinect_post_path + 'depth_' + '{0:04}'.format(count) + '_' + img_name + '.jpg', depth_img)
                new_skele1.append(skele1[i])
                new_skele2.append(skele2[i])
                if save_add_path:
                    cv2.imwrite(kinect_post_path2 + '{0:04}'.format(count) + '_' + img_name + '.jpg', color_img)
                count += 1

            with open(kinect_post_path + 'skele1.p', 'wb') as f:
                pickle.dump(new_skele1, f)
            with open(kinect_post_path + 'skele2.p', 'wb') as f:
                pickle.dump(new_skele2, f)

            print("dumping tracker")
            gazes_new = []
            count = 0
            if save_add_path:
                tracker_post_path2 = './annotations/' + clip_name + '/tracker/'
                if not os.path.exists(tracker_post_path2):
                    os.makedirs(tracker_post_path2)
            for i in range(tracker_start_frame, tracker_end_frame):
                img = cv2.imread(eye_tracker_color_frames[i])
                img_name = eye_tracker_color_frames[i].split('/')[-1].split('.')[0]
                gaze = gazes[i]
                if not np.isnan(gaze['gaze'][0]) and not np.isnan(gaze['gaze'][1]):
                    cv2.circle(img, (int(gaze['gaze'][0]), int(gaze['gaze'][1])), 7, (255, 0, 255), thickness=5)
                cv2.imwrite(tracker_post_path + '{0:04}'.format(count) + '_' + img_name + '.jpg', img)
                gazes_new.append(gazes[i])
                if save_add_path:
                    cv2.imwrite(tracker_post_path2 + '{0:04}'.format(count) + '_' + img_name + '.jpg', img)
                count += 1

            with open(gaze_post_path + '.p', 'wb') as f:
                pickle.dump(gazes_new, f)








