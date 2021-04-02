import os
import pickle

data_path = './published_frames'
folders = os.listdir(data_path)
sorted_folders = sorted(folders)
print(sorted_folders)
f = open('sorted_list.p', 'wb')
pickle.dump(sorted_folders, f)