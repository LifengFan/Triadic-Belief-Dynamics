import pickle
import cPickle
import os
from os import path, listdir
import numpy as np

root='/home/lfan/Dropbox/Projects/ECCV20/'
#------------------------------------------------
# #video
# [['first false belief', 22],
# ['joint attention', 16],
# ['individual', 23],
# ['gaze follow', 18],
# ['second false belief', 15]]


# #frame
# {'first false belief': 28243,
# 'joint attention': 12868,
# 'individual': 28806,
# 'gaze follow': 21030,
# 'second false belief': 18384}

#-------------------------------------------------
# file folder name mapping

# folders=sorted(listdir(path.join(root,'post_skeletons')))
# print(folders)
#
# folder_name_map={}
#
# cnt=0
# for f in folders:
#     cnt+=1
#     folder_name_map[str(cnt)]=f
#
# print(folder_name_map)

# folder_name_map={'24': 'test_94342_26', '25': 'test_94342_3', '26': 'test_94342_4', '27': 'test_94342_5',
#                  '20': 'test_94342_22', '21': 'test_94342_23', '22': 'test_94342_24', '23': 'test_94342_25',
#                  '28': 'test_94342_6', '29': 'test_94342_7', '4': 'test7', '8': 'test_94342_11',
#                  '59': 'test_boelter3_8', '58': 'test_boelter3_7', '55': 'test_boelter3_4',
#                  '54': 'test_boelter3_3', '57': 'test_boelter3_6', '56': 'test_boelter3_5',
#                  '51': 'test_boelter3_12', '50': 'test_boelter3_11', '53': 'test_boelter3_2',
#                  '52': 'test_boelter3_13', '88': 'test_boelter_25', '89': 'test_boelter_3',
#                  '82': 'test_boelter_18', '83': 'test_boelter_19', '80': 'test_boelter_15',
#                  '81': 'test_boelter_17', '86': 'test_boelter_22', '87': 'test_boelter_24',
#                  '84': 'test_boelter_2', '85': 'test_boelter_21', '3': 'test6', '7': 'test_94342_10',
#                  '39': 'test_boelter2_17', '38': 'test_boelter2_16', '33': 'test_9434_3',
#                  '32': 'test_9434_18', '31': 'test_9434_1', '30': 'test_94342_8', '37': 'test_boelter2_15',
#                  '36': 'test_boelter2_14', '35': 'test_boelter2_12', '34': 'test_boelter2_0',
#                  '60': 'test_boelter3_9', '61': 'test_boelter4_0', '62': 'test_boelter4_1',
#                  '63': 'test_boelter4_10', '64': 'test_boelter4_11', '65': 'test_boelter4_12',
#                  '66': 'test_boelter4_13', '67': 'test_boelter4_2', '68': 'test_boelter4_3',
#                  '69': 'test_boelter4_4', '2': 'test2', '6': 'test_94342_1', '91': 'test_boelter_5',
#                  '90': 'test_boelter_4', '93': 'test_boelter_7', '92': 'test_boelter_6', '94': 'test_boelter_9',
#                  '11': 'test_94342_14', '10': 'test_94342_13', '13': 'test_94342_16', '12': 'test_94342_15',
#                  '15': 'test_94342_18', '14': 'test_94342_17', '17': 'test_94342_2', '16': 'test_94342_19',
#                  '19': 'test_94342_21', '18': 'test_94342_20', '48': 'test_boelter3_1', '49': 'test_boelter3_10',
#                  '46': 'test_boelter2_8', '47': 'test_boelter3_0', '44': 'test_boelter2_6', '45': 'test_boelter2_7',
#                  '42': 'test_boelter2_4', '43': 'test_boelter2_5', '40': 'test_boelter2_2', '41': 'test_boelter2_3',
#                  '1': 'test1', '5': 'test_94342_0', '9': 'test_94342_12', '77': 'test_boelter_12',
#                  '76': 'test_boelter_10', '75': 'test_boelter_1', '74': 'test_boelter4_9', '73': 'test_boelter4_8',
#                  '72': 'test_boelter4_7', '71': 'test_boelter4_6', '70': 'test_boelter4_5', '79': 'test_boelter_14', '78': 'test_boelter_13'}


annot_map={'test1': 'Task_0_1.txt', 'test2': 'Task_3_1.txt', 'test7': 'Task_9_1.txt', 'test6': 'Task_6_1.txt',
           'test_boelter2_8': 'Task_174_1.txt', 'test_boelter2_6': 'Task_168_1.txt', 'test_boelter2_7': 'Task_171_1.txt',
           'test_boelter2_3': 'Task_153_1.txt', 'test_boelter2_0': 'Task_126_1.txt', 'test_boelter2_12': 'Task_129_1.txt',
           'test_boelter2_14': 'Task_132_1.txt', 'test_boelter2_15': 'Task_135_1.txt', 'test_boelter2_16': 'Task_138_1.txt',
           'test_boelter2_17': 'Task_141_1.txt', 'test_94342_26': 'Task_72_1.txt', 'test_94342_25': 'Task_69_1.txt',
           'test_94342_24': 'Task_66_1.txt', 'test_94342_23': 'Task_63_1.txt', 'test_94342_22': 'Task_60_1.txt',
           'test_94342_21': 'Task_57_1.txt', 'test_94342_20': 'Task_54_1.txt', 'test_9434_18': 'Task_12_1.txt',
           'test_boelter_21': 'Task_144_1.txt', 'test_boelter4_8': 'Task_261_1.txt', 'test_boelter4_13': 'Task_237_1.txt',
           'test_boelter4_10': 'Task_228_1.txt', 'test_boelter4_11': 'Task_231_1.txt', 'test_boelter4_4': 'Task_249_1.txt',
           'test_boelter4_5': 'Task_252_1.txt', 'test_boelter4_6': 'Task_255_1.txt', 'test_boelter4_7': 'Task_258_1.txt',
           'test_boelter4_0': 'Task_225_1.txt', 'test_boelter4_1': 'Task_240_1.txt', 'test_boelter4_2': 'Task_243_1.txt',
           'test_boelter4_3': 'Task_246_1.txt', 'test_boelter_19': 'Task_120_1.txt', 'test_boelter_18': 'Task_117_1.txt',
           'test_boelter_10': 'Task_99_1.txt', 'test_boelter_13': 'Task_105_1.txt', 'test_boelter_12': 'Task_102_1.txt',
           'test_boelter_15': 'Task_111_1.txt', 'test_boelter_14': 'Task_108_1.txt', 'test_boelter_17': 'Task_114_1.txt',
           'test_boelter4_12': 'Task_234_1.txt', 'test_boelter4_9': 'Task_264_1.txt', 'test_9434_3': 'Task_96_1.txt',
           'test_9434_1': 'Task_15_1.txt', 'test_boelter3_13': 'Task_192_1.txt', 'test_boelter3_12': 'Task_189_1.txt',
           'test_boelter3_11': 'Task_186_1.txt', 'test_boelter3_10': 'Task_183_1.txt', 'test_boelter_9': 'Task_279_1.txt',
           'test_boelter_5': 'Task_270_1.txt', 'test_boelter_4': 'Task_267_1.txt', 'test_boelter_7': 'Task_276_1.txt',
           'test_boelter_6': 'Task_273_1.txt', 'test_boelter_1': 'Task_123_1.txt', 'test_boelter_3': 'Task_222_1.txt',
           'test_boelter_2': 'Task_177_1.txt', 'test_94342_8': 'Task_93_1.txt', 'test_94342_0': 'Task_18_1.txt',
           'test_94342_1': 'Task_51_1.txt', 'test_94342_2': 'Task_75_1.txt', 'test_94342_3': 'Task_78_1.txt',
           'test_94342_4': 'Task_81_1.txt', 'test_94342_5': 'Task_84_1.txt', 'test_94342_6': 'Task_87_1.txt',
           'test_94342_7': 'Task_90_1.txt', 'test_boelter3_7': 'Task_213_1.txt', 'test_boelter3_6': 'Task_210_1.txt',
           'test_boelter3_5': 'Task_207_1.txt', 'test_boelter3_4': 'Task_204_1.txt', 'test_boelter3_3': 'Task_201_1.txt',
           'test_boelter3_2': 'Task_198_1.txt', 'test_boelter3_1': 'Task_195_1.txt', 'test_boelter3_0': 'Task_180_1.txt',
           'test_boelter3_9': 'Task_219_1.txt', 'test_boelter3_8': 'Task_216_1.txt', 'test_94342_12': 'Task_27_1.txt',
           'test_94342_13': 'Task_30_1.txt', 'test_94342_10': 'Task_21_1.txt', 'test_94342_11': 'Task_24_1.txt',
           'test_94342_16': 'Task_39_1.txt', 'test_94342_17': 'Task_42_1.txt', 'test_94342_14': 'Task_33_1.txt',
           'test_94342_15': 'Task_36_1.txt', 'test_94342_18': 'Task_45_1.txt', 'test_94342_19': 'Task_48_1.txt'}

# training, validation, test -- split

def color_bank(id):

    bank=["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
     "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
     "#8dd3c7", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5",
     "#ffed6f", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999", "#621e15",
     "#e59076", "#128dcd", "#083c52", "#64c5f2", "#61afaf", "#0f7369", "#9c9da1", "#365e96", "#983334", "#77973d",
     "#5d437c", "#36869f", "#d1702f", "#8197c5", "#c47f80", "#acc484", "#9887b0", "#2d588a", "#58954c", "#e9a044",
     "#c12f32", "#723e77", "#7d807f", "#9c9ede", "#7375b5", "#4a5584", "#cedb9c", "#b5cf6b", "#8ca252", "#637939",
     "#e7cb94", "#e7ba52", "#bd9e39", "#8c6d31", "#e7969c", "#d6616b", "#ad494a", "#843c39", "#de9ed6", "#ce6dbd",
     "#a55194", "#7b4173", "#000000", "#0000FF"]

    L=len(bank)

    return bank[id%L]


#--------------------------------------------------------------------------------------------------------

num_video=94

#----------------------------------------------------------------------------------------------------------
# subevent annotation

## SubEvent ID: no_communication 0, attention_follow 1, joint_attention 2, mutual 3,

# tracker--complete annottion
# battery--only no_communication segment annotation


# event_seg_tracker= {'test_9434_1': [[0, 57, 0], [58, 124, 0], [125, 182, 1], [183, 251, 2], [252, 417, 0]],
#                     'test_94342_16':[[0, 21, 0], [22, 45, 0], [46, 84, 0], [85, 158, 1], [159, 200, 1], [201, 214, 0],
#                                      [215, 370, 1], [371, 524, 1], [525, 587, 3], [588, 782, 2], [783, 1009, 2]],
#                     'test_boelter4_12': [[0,141,0], [142, 462, 2], [463, 605, 0], [606, 942, 2], [943, 1232, 2], [1233, 1293, 0]],
#                     'test_boelter4_9':[[0,27,0], [28, 172, 0], [173, 221, 0], [222, 307, 1], [308, 466, 0], [467, 794, 1], [795, 866, 1],
#                                        [867, 1005, 2], [1006, 1214, 2], [1215, 1270, 0]],
#                     'test_boelter4_4':[[0, 120, 0],[121, 183, 0], [184, 280, 1], [281, 714, 0]],
#                     'test_boelter4_3':[[0,117, 0], [118, 200, 1], [201, 293, 1], [294, 404, 1], [405, 600, 1], [601, 800, 1], [801, 905, 1],
#                                        [906, 1234, 1]],
#                     'test_boelter4_1':[[0,310, 0], [311, 560, 0], [561, 680, 0], [681, 748, 0], [749, 839, 0], [840, 1129, 0], [1130, 1237, 0]],
#                     'test_boelter3_13':[[0, 204, 2], [205, 300, 2], [301, 488, 2], [489, 755, 2]],
#                     'test_boelter3_11':[[0,254,1], [255, 424, 0], [425, 598, 1], [599, 692, 0], [693, 772, 2], [773, 878, 2], [879, 960, 2], [961, 1171, 2],
#                                         [1172, 1397, 2]],
#                     'test_boelter3_6':[[0, 174, 1], [175, 280, 1], [281, 639, 0], [640, 695, 1], [696, 788, 0], [789, 887, 2], [888, 1035, 1], [1036, 1445, 2]],
#                     'test_boelter3_4':[[0, 158, 1], [159, 309, 1], [310, 477, 1], [478, 668, 1], [669, 780, 1], [781, 817, 0], [818, 848, 1], [849, 942, 1]],
#                     'test_boelter3_0':[[0, 140, 0], [141, 353, 0], [354, 599, 0], [600, 727, 0], [728, 768, 0]],
#                     'test_boelter2_15': [[0, 46, 0], [47, 252, 2], [253, 298, 1], [299, 414, 2], [415, 547, 2], [548, 690, 1], [691, 728, 1], [729, 773, 2],
#                                          [774, 935, 2]],
#                     'test_boelter2_12':[[0, 163, 0], [164, 285, 1], [286, 444, 1], [445, 519, 0], [520, 583, 1], [584, 623, 0], [624, 660, 0],
#                                         [661, 854, 1], [855, 921, 1], [922, 1006, 2], [1007, 1125, 2], [1126, 1332, 2], [1333, 1416, 2]],
#                     'test_boelter2_5':[[0,94, 0], [95, 176, 1], [177, 246, 1], [247, 340, 1], [341, 442, 1], [443, 547, 1], [548, 654, 1], [655, 734, 0],
#                                        [735, 792, 0], [793, 1019, 0], [1020, 1088, 0], [1089, 1206, 0], [1207, 1316, 1], [1317, 1466, 1], [1467, 1787, 2],
#                                        [1788, 1936, 1], [1937, 2084, 2]],
#                     'test_boelter2_4':[[0, 260, 1], [261, 421, 1], [422, 635, 1], [636, 741, 1], [742, 846, 1], [847, 903, 1], [904, 953, 1], [954, 1005, 1],
#                                        [1006, 1148, 1], [1149, 1270, 1], [1271, 1525, 1]],
#                     'test_boelter2_2':[[0, 131, 0],[132, 226, 0], [227, 267, 0], [268, 352, 0], [353, 412, 0], [413, 457, 0], [458, 502, 0],
#                                        [503, 532, 0], [533, 578, 0], [579, 640, 0], [641, 722, 0], [723, 826, 0], [827, 913, 0], [914, 992, 0],
#                                        [993, 1070, 0], [1071, 1265, 0], [1266, 1412, 0]],
#                     'test_boelter_21':[[0, 238, 1],[239, 310, 0], [311, 373, 1], [374, 457, 0], [458, 546, 3], [547, 575, 1],
#                                        [576, 748, 2], [749, 952, 2]],
#                     }
#
# event_seg_battery= {'test_9434_1': [[0,67,0], [68, 124, 0], [252, 343, 0], [344, 380, 0], [381, 417, 0]],
#                     'test_94342_16': [[0,84,0], [201, 214, 0]],
#                     'test_boelter4_12':[[0,32,0], [33, 141,0], [463, 519,0], [520, 597, 0], [598, 605, 0], [1233, 1293,0]],
#                     'test_boelter4_9':[[0, 221, 0], [308, 466, 0], [1215, 1270, 0]],
#                     'test_boelter4_4':[[0,183, 0],[281, 529, 0], [530,714, 0]],
#                     'test_boelter4_3':[[0,117,0]],
#                     'test_boelter4_1':[[0, 252, 0], [253, 729, 0], [730, 1202, 0], [1203, 1237, 0]],
#                     'test_boeler3_13':[],
#                     'test_boelter3_11':[[255, 424, 0], [599, 692, 0]],
#                     'test_boelter3_6': [[281, 498, 0], [499, 639, 0], [696, 748, 0], [749, 788, 0]],
#                     'test_boelter3_4': [[781, 817, 0]],
#                     'test_boelter3_0': [[0, 102, 0], [103, 480, 0], [481, 703, 0], [704, 768, 0]],
#                     'test_boelter2_15':[[0, 46, 0]],
#                     'test_boelter2_12':[[0, 163, 0], [445, 519, 0], [584, 660, 0]],
#                     'test_boelter2_5':[[0, 94, 0], [655, 1206, 0]],
#                     'test_boelter2_4': [],
#                     'test_boelter2_2':[[0, 145, 0], [146, 224, 0], [225, 271, 0], [272, 392, 0], [393, 454, 0],
#                                        [455, 762, 0], [763, 982, 0], [983, 1412, 0]],
#                     'test_boelter_21':[[239, 285, 0], [286, 310, 0], [374, 457, 0]],
#                      }


### annotated by Lifeng

# event_seg_tracker = {'test_9434_18': [[0, 749, 0], [750, 824, 0], [825, 863, 2], [864, 974, 0], [975, 1041, 0]],
#                              'test_94342_1': [[0, 13, 0], [14, 104, 0], [105, 333, 0], [334, 451, 0], [452, 652, 0],
#                                               [653, 897, 0], [898, 1076, 0], [1077, 1181, 0], [1181, 1266, 0],
#                                               [1267, 1386, 0]],
#                              'test_94342_6': [[0, 95, 0], [96, 267, 1], [268, 441, 1], [442, 559, 1], [560, 681, 1], [
#             682, 796, 1], [797, 835, 1], [836, 901, 0], [902, 943, 1]],
#         'test_94342_10': [[0, 36, 0], [37, 169, 0], [170, 244, 1], [245, 424, 0], [425, 599, 0], [600, 640, 0],
#                           [641, 680, 0], [681, 726, 1], [727, 866, 2], [867, 1155, 2]],
#         'test_94342_21': [[0, 13, 0], [14, 66, 3], [67, 594, 2], [595, 1097, 2], [1098, 1133, 0]],
#         'test1': [[0, 477, 0], [478, 559, 0], [560, 689, 2], [690, 698, 0]],
#         'test6': [[0, 140, 0], [141, 375, 0], [375, 678, 0], [679, 703, 0]],
#         'test7': [[0, 100, 0], [101, 220, 2], [221, 226, 0]],
#         'test_boelter_2': [[0, 154, 0], [155, 279, 0], [280, 371, 0], [372, 450, 0], [451, 470, 0], [471, 531, 0],
#                            [532, 606, 0]],
#         'test_boelter_7': [[0, 69, 0], [70, 118, 1], [119, 239, 0], [240, 328, 1], [329, 376, 0], [377, 397, 1],
#                            [398, 520, 0], [521, 564, 0], [565, 619, 1], [620, 688, 1], [689, 871, 0], [872, 897, 0],
#                            [898, 958, 1], [959, 1010, 0], [1011, 1084, 0], [1085, 1140, 0], [1141, 1178, 0],
#                            [1179, 1267, 1], [1268, 1317, 0], [1318, 1327, 0]],
#         'test_boelter_24': [[0, 62, 0], [63, 185, 2], [186, 233, 2], [234, 292, 2], [293, 314, 0]],
#         'test_boelter_12': [[0, 47, 1], [48, 119, 0], [120, 157, 1], [158, 231, 0], [232, 317, 0], [318, 423, 0], [424,
#                                                                                                                   459,
#                                                                                                                   0], [
#                                            460, 522, 0], [523, 586, 0], [587, 636, 0], [637, 745, 1], [746, 971, 2]],
#         'test_9434_1': [[0, 57, 0], [58, 124, 0], [125, 182, 1], [183, 251, 2],
#                       [252, 417, 0]],
#         'test_94342_16': [[0, 21, 0], [22, 45, 0], [46, 84, 0], [85, 158, 1], [159, 200, 1],
#                         [201, 214, 0],
#                         [215, 370, 1], [371, 524, 1], [525, 587, 3], [588, 782, 2],
#                         [783, 1009, 2]],
#         'test_boelter4_12': [[0, 141, 0], [142, 462, 2], [463, 605, 0], [606, 942, 2],
#                            [943, 1232, 2], [1233, 1293, 0]],
#         'test_boelter4_9': [[0, 27, 0], [28, 172, 0], [173, 221, 0], [222, 307, 1],
#                           [308, 466, 0], [467, 794, 1], [795, 866, 1],
#                           [867, 1005, 2], [1006, 1214, 2], [1215, 1270, 0]],
#         'test_boelter4_4': [[0, 120, 0], [121, 183, 0], [184, 280, 1], [281, 714, 0]],
#         'test_boelter4_3': [[0, 117, 0], [118, 200, 1], [201, 293, 1], [294, 404, 1],
#                           [405, 600, 1], [601, 800, 1], [801, 905, 1],
#                           [906, 1234, 1]],
#         'test_boelter4_1': [[0, 310, 0], [311, 560, 0], [561, 680, 0], [681, 748, 0],
#                           [749, 839, 0], [840, 1129, 0], [1130, 1237, 0]],
#         'test_boelter3_13': [[0, 204, 2], [205, 300, 2], [301, 488, 2], [489, 755, 2]],
#         'test_boelter3_11': [[0, 254, 1], [255, 424, 0], [425, 598, 1], [599, 692, 0],
#                            [693, 772, 2], [773, 878, 2], [879, 960, 2], [961, 1171, 2],
#                            [1172, 1397, 2]],
#         'test_boelter3_6': [[0, 174, 1], [175, 280, 1], [281, 639, 0], [640, 695, 1],
#                           [696, 788, 0], [789, 887, 2], [888, 1035, 1], [1036, 1445, 2]],
#         'test_boelter3_4': [[0, 158, 1], [159, 309, 1], [310, 477, 1], [478, 668, 1],
#                           [669, 780, 1], [781, 817, 0], [818, 848, 1], [849, 942, 1]],
#         'test_boelter3_0': [[0, 140, 0], [141, 353, 0], [354, 599, 0], [600, 727, 0],
#                           [728, 768, 0]],
#         'test_boelter2_15': [[0, 46, 0], [47, 252, 2], [253, 298, 1], [299, 414, 2],
#                            [415, 547, 2], [548, 690, 1], [691, 728, 1], [729, 773, 2],
#                            [774, 935, 2]],
#         'test_boelter2_12': [[0, 163, 0], [164, 285, 1], [286, 444, 1], [445, 519, 0],
#                            [520, 583, 1], [584, 623, 0], [624, 660, 0],
#                            [661, 854, 1], [855, 921, 1], [922, 1006, 2], [1007, 1125, 2],
#                            [1126, 1332, 2], [1333, 1416, 2]],
#         'test_boelter2_5': [[0, 94, 0], [95, 176, 1], [177, 246, 1], [247, 340, 1],
#                           [341, 442, 1], [443, 547, 1], [548, 654, 1], [655, 734, 0],
#                           [735, 792, 0], [793, 1019, 0], [1020, 1088, 0], [1089, 1206, 0],
#                           [1207, 1316, 1], [1317, 1466, 1], [1467, 1787, 2],
#                           [1788, 1936, 1], [1937, 2084, 2]],
#         'test_boelter2_4': [[0, 260, 1], [261, 421, 1], [422, 635, 1], [636, 741, 1],
#                           [742, 846, 1], [847, 903, 1], [904, 953, 1], [954, 1005, 1],
#                           [1006, 1148, 1], [1149, 1270, 1], [1271, 1525, 1]],
#         'test_boelter2_2': [[0, 131, 0], [132, 226, 0], [227, 267, 0], [268, 352, 0],
#                           [353, 412, 0], [413, 457, 0], [458, 502, 0],
#                           [503, 532, 0], [533, 578, 0], [579, 640, 0], [641, 722, 0],
#                           [723, 826, 0], [827, 913, 0], [914, 992, 0],
#                           [993, 1070, 0], [1071, 1265, 0], [1266, 1412, 0]],
#         'test_boelter_21': [[0, 238, 1], [239, 310, 0], [311, 373, 1], [374, 457, 0],
#                           [458, 546, 3], [547, 575, 1],
#                           [576, 748, 2], [749, 952, 2]],
#         }
#
#
# event_seg_battery = {
#             'test_9434_18': [[0, 96, 0], [97, 361, 0], [362, 528, 0], [529, 608, 0], [609, 824, 0], [864, 1041, 0]],
#             'test_94342_1': [[0, 751, 0], [752, 876, 0], [877, 1167, 0], [1168, 1386, 0]],
#             'test_94342_6': [[0, 95, 0], [836, 901, 0]],
#             'test_94342_10': [[0, 156, 0], [157, 169, 0], [245, 274, 0], [275, 389, 0], [390, 525, 0], [526, 665, 0],
#                               [666, 680, 0]],
#             'test_94342_21': [[0, 13, 0], [1098, 1133, 0]],
#             'test1': [[0, 94, 0], [95, 155, 0], [156, 225, 0], [226, 559, 0], [690, 698, 0]],
#             'test6': [[0, 488, 0], [489, 541, 0], [542, 672, 0], [672, 803, 0]],
#             'test7': [[0, 70, 0], [71, 100, 0], [221, 226, 0]],
#             'test_boelter_2': [[0, 318, 0], [319, 458, 0], [459, 543, 0], [544, 606, 0]],
#             'test_boelter_7': [[0, 69, 0], [119, 133, 0], [134, 187, 0], [188, 239, 0], [329, 376, 0], [398, 491, 0],
#                                [492, 564, 0], [689, 774, 0], [775, 862, 0], [863, 897, 0], [959, 1000, 0],
#                                [1001, 1178, 0], [1268, 1307, 0], [1307, 1327, 0]],
#             'test_boelter_24': [[0, 62, 0], [293, 314, 0]],
#             'test_boelter_12': [[48, 219, 0], [220, 636, 0]],
#             'test_9434_1': [[0, 67, 0], [68, 124, 0], [252, 343, 0], [344, 380, 0], [381, 417, 0]],
#             'test_94342_16': [[0, 84, 0], [201, 214, 0]],
#             'test_boelter4_12': [[0, 32, 0], [33, 141, 0], [463, 519, 0], [520, 597, 0], [598, 605, 0],
#                                  [1233, 1293, 0]],
#             'test_boelter4_9': [[0, 221, 0], [308, 466, 0], [1215, 1270, 0]],
#             'test_boelter4_4': [[0, 183, 0], [281, 529, 0], [530, 714, 0]],
#             'test_boelter4_3': [[0, 117, 0]],
#             'test_boelter4_1': [[0, 252, 0], [253, 729, 0], [730, 1202, 0], [1203, 1237, 0]],
#             'test_boelter3_13': [],
#             'test_boelter3_11': [[255, 424, 0], [599, 692, 0]],
#             'test_boelter3_6': [[281, 498, 0], [499, 639, 0], [696, 748, 0], [749, 788, 0]],
#             'test_boelter3_4': [[781, 817, 0]],
#             'test_boelter3_0': [[0, 102, 0], [103, 480, 0], [481, 703, 0], [704, 768, 0]],
#             'test_boelter2_15': [[0, 46, 0]],
#             'test_boelter2_12': [[0, 163, 0], [445, 519, 0], [584, 660, 0]],
#             'test_boelter2_5': [[0, 94, 0], [655, 1206, 0]],
#             'test_boelter2_4': [],
#             'test_boelter2_2': [[0, 145, 0], [146, 224, 0], [225, 271, 0], [272, 392, 0], [393, 454, 0],
#                                 [455, 762, 0], [763, 982, 0], [983, 1412, 0]],
#             'test_boelter_21': [[239, 285, 0], [286, 310, 0], [374, 457, 0]],
#         }


# event_seg_battery_new = {}
#
# for key, item in event_seg_tracker.items():
#     item = np.array(item)
#     item1 = item[item[:, 2] == 1]
#     item2 = item[item[:, 2] == 2]
#     item3 = item[item[:, 2] == 3]
#     total = np.vstack([item1, item2, item3])
#     item_b = event_seg_battery[key]
#     item_b = np.array(item_b)
#     if item_b.shape[0] == 0:
#         item_b_new = total
#     else:
#         item_b_new = np.vstack([item_b, total])
#     item_b_idx = np.argsort(item_b_new[:, 0])
#     item_b_sort = item_b_new[item_b_idx].tolist()
#     event_seg_battery_new[key] = item_b_sort
#
#
# with open('event_seg_battery', 'wb') as filehandle:
#     pickle.dump(event_seg_battery_new, filehandle)
#
# with open('event_seg_tracker', 'wb') as filehandle:
#     pickle.dump(event_seg_tracker, filehandle)
#
# pass

trackers_type = {'test1': 'skele1.p', 'test2': 'skele2.p', 'test6': 'skele2.p', 'test7': 'skele1.p',
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















