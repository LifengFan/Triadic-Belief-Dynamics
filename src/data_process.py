import cv2
import numpy as np
import os
from os import listdir, path
import pickle
from PIL import Image, ImageFont
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

data_path='/home/lfan/Dropbox/Projects/CVPR20/data/tracker/'

#----------------------------------------------------------------------------
# split videos into images and save them into a folder with the same name

# for sub_d in listdir(data_path):
#
#     if path.exists(path.join(data_path,sub_d)) and path.isdir(path.join(data_path,sub_d)):
#
#         file_list=listdir(path.join(data_path,sub_d))
#
#         for f in file_list:
#             if path.isfile(path.join(data_path,sub_d,f)) and f.endswith('.avi'):
#
#                 if not path.exists(path.join(data_path,sub_d,'img_'+f.rstrip('.avi'))):
#                     os.mkdir(path.join(data_path,sub_d,'img_'+f.rstrip('.avi')))
#
#                 print('ffmpeg -i '+path.join(data_path,sub_d,f)+' '+path.join(data_path,sub_d, 'img_'+f.rstrip('.avi'))+'/%05d.png ')
#                 os.system('ffmpeg -i '+path.join(data_path,sub_d,f)+' '+path.join(data_path,sub_d, 'img_'+f.rstrip('.avi'))+'/%05d.png ')

#---------------------------------------------------------------------------------------------------------------
# read gaze position info from raw annotation files and save into python numpy array

# for sub_d in listdir(data_path):
#
#      if path.exists(path.join(data_path,sub_d)) and path.isdir(path.join(data_path,sub_d)):
#
#          file_list = listdir(path.join(data_path, sub_d))
#
#          for f in file_list:
#              if path.isfile(path.join(data_path, sub_d, f)) and f.endswith('.txt'):
#
#                    f_reader=open(path.join(data_path,sub_d,f),'r')
#                    lines=f_reader.readlines()
#
#                    cnt=0
#                    for line in lines:
#
#                        if line.startswith('Time'):
#                            print(cnt)
#                            #print(lines[cnt])
#                            break
#                        cnt+=1
#
#                    lines_comp=lines[(cnt+1):]
#
#                    gaze_pos_list=[]
#
#                    cnt=0
#                    for line in lines_comp:
#
#                        tmp=line.rstrip('\n').split('\t')
#
#                        gaze_pos_list.append({'id':cnt,'time_machine':tmp[0],'time_real':tmp[5],'gaze':[float(tmp[3]),float(tmp[4])]})
#
#                        cnt+=1
#
#
#                    save_name=path.join(data_path,sub_d,f.rstrip('.txt')+'.data')
#                    with open(save_name, 'wb') as filehandle:
#
#                        pickle.dump(gaze_pos_list, filehandle)

# #---------------------------------------------------------------------------------------------
# # verify that the gaze position can be mapped to videos
#
# imgs=sorted(listdir(path.join(data_path, 'Yining-[51e5aab8-07ae-4038-b7b4-b2dd3730f036]','img_Yining-1-recording')))
#
# with open(path.join(data_path, 'Yining-[51e5aab8-07ae-4038-b7b4-b2dd3730f036]','1.data'),'rb') as filehandle:
#
#     gaze=pickle.load(filehandle)
#
#
# L_img=len(imgs)
# L_gaze=len(gaze)
#
# print(L_img)
# print(L_gaze)
#
# #gaze_index=sorted(np.random.choice(L_gaze, L_img, replace=False))
#
# gaze_index=[]
#
# r=(L_gaze-1)/(L_img-1)
#
# for i in range(L_img):
#
#     gaze_index.append(round(i*r))
#
#
# for i in range(1,L_img,100):
#
#     img = np.array(Image.open(path.join(data_path, 'Yining-[51e5aab8-07ae-4038-b7b4-b2dd3730f036]','img_Yining-1-recording',imgs[i])), dtype=np.uint8)
#
#     fig, ax = plt.subplots(1)
#     ax.imshow(img)
#
#     # draw human bbx
#     gaze_now=gaze[gaze_index[i]]['gaze']
#
#     #rect = patches.Circle(xy=gaze_now, radius=5)
#     #ax.add_patch(rect)
#
#     ax.plot(gaze_now[0],gaze_now[1],'r.', markersize=12)
#
#     plt.axis('off')
#     plt.margins(0, 0)
#     plt.savefig(path.join(data_path,'Yining-[51e5aab8-07ae-4038-b7b4-b2dd3730f036]','gaze_verify', imgs[i]), dpi=1000)
#

#-----------------------------------------------------------------------------------------------------------------
# generate new gaze position file

for sub_d in listdir(data_path):

    if path.exists(path.join(data_path, sub_d)) and path.isdir(path.join(data_path, sub_d)):

        file_list= listdir(path.join(data_path, sub_d))

        for d in file_list:

            if path.isdir(path.join(data_path, sub_d, d)) and d.startswith('img_'):

               ID=d.split('-')[1]

               print(ID)

               with open(path.join(data_path, sub_d, ID+'.data'), 'rb') as filehandle:
                   gaze=pickle.load(filehandle)


               L_img=len(listdir(path.join(data_path, sub_d, d)))
               L_gaze=len(gaze)

               print(L_img)
               print(L_gaze)

               gaze_index=[]

               r=(L_gaze-1)/(L_img-1)

               for i in range(L_img):
                   gaze_index.append(round(i*r))


               gaze_new=[]
               for j in range(L_img):

                   ind=gaze_index[j]
                   gaze_new.append(gaze[ind])

               with open(path.join(data_path, sub_d, 'new_'+ ID +'.data'), 'wb') as filehandle:
                   pickle.dump(gaze_new, filehandle)

