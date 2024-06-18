import os
import numpy as np
import shutil

origin_path = '/home/jhon/Downloads/train_set/annotations/'
origin_path_img = '/home/jhon/Downloads/train_set/images/'
dest_path = '/home/jhon/Downloads/AffectNet/'

items = os.listdir(origin_path)
items = sorted(items)

for i in items:
    if 'exp' in i:
        print('\tProcessing: ', i)
        exp = np.load('/home/jhon/Downloads/train_set/annotations/' + i)
        name = i[:i.find('_')] + '.jpg'
        if int(exp.item()) == 0:
            shutil.copy(origin_path_img + name, dest_path + '/0/' + name)
        elif int(exp.item()) == 1:
            shutil.copy(origin_path_img + name, dest_path + '/1/' + name)
        elif int(exp.item()) == 2:
            shutil.copy(origin_path_img + name, dest_path + '/2/' + name)
        elif int(exp.item()) == 3:
            shutil.copy(origin_path_img + name, dest_path + '/3/' + name)
        elif int(exp.item()) == 4:
            shutil.copy(origin_path_img + name, dest_path + '/4/' + name)
        elif int(exp.item()) == 5:
            shutil.copy(origin_path_img + name, dest_path + '/5/' + name)
        elif int(exp.item()) == 6:
            shutil.copy(origin_path_img + name, dest_path + '/6/' + name)
        elif int(exp.item()) == 7:
            shutil.copy(origin_path_img + name, dest_path + '/7/' + name)
        else:
            print('Not found emotion for: ', origin_path + name)

