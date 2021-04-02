import os
from os import listdir
import random

path='/home/lfan/Dropbox/Projects/NIPS20/supplementary/fig_dataset_diversity'
destination='/home/lfan/Dropbox/Projects/NIPS20/supplementary/dataset_diversity'
imgs=listdir(path)

random.shuffle(imgs)

cnt=0
for img in imgs:
    cnt+=1
    print(cnt)
    os.system('cp '+os.path.join(path, img)+ ' '+ os.path.join(destination, str(cnt)+'.jpg'))


