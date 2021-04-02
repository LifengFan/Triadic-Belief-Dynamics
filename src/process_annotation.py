from os import path, listdir
import numpy as np
from metadata import folder_name_map
import pickle

# with open('id_map.txt', 'rb') as f:
#     lines=f.readlines()
#
#
# annot_map={}
#
# for line in lines:
#     id=int(line.split("\t")[0])
#     folder=line.split("\t")[1].split('\n')[0]
#     annot_file='Task_'+str(id-1)+'_1.txt'
#
#     annot_map[folder]=annot_file
#
# to_write=open('annot_map.pkl','wb')
# pickle.dump(annot_map, to_write)
# to_write.close()

# task_id=0
# for folder in folders:
#     print task_id, folder
#
#     task_id+=3
#
# pass

# for folder_id in range(len(sorted_list)):
#
#     folder=sorted_list[folder_id]
#
#     if folder.endswith('_kinect'):
#
#         task_id=folder_id-1
#
#         print folder, 'Task_'+str(task_id)+'_1'


# folders=sorted(listdir('/home/lfan/Dropbox/Projects/ECCV20/annotations/'))
#
# annot_dict={}
#
# for f_id in range(len(folders)):
#
#     task_id=3*f_id
#     annot_file = 'Task_' + str(task_id) + '_1.txt'
#
#     print f_id, folders[f_id], task_id, annot_file
#
#     with open(path.join('/home/lfan/Dropbox/Projects/ECCV20/annot/all/', annot_file), 'rb') as to_read:
#
#         annot_dict[folders[f_id]]=to_read.readlines()
#
#
# to_write=open('annot_dict.pkl', 'wb')
# pickle.dump(annot_dict, to_write)
# to_write.close()

# # check the annotation file mapping
#
# file=open('annot_dict.pkl', 'rb')
#
# annot_dict=pickle.load(file)
# file.close()
#
#
# folder='test1'
#
# annot=annot_dict[folder]


#---------------------------------------------------------------------------------------------

to_read=open('annot_map.pkl', 'rb')

annot_map=pickle.load(to_read)

print(annot_map)
















