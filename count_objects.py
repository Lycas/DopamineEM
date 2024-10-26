import fnmatch
import math
import tifffile
import numpy as np
from scipy.ndimage import label
import pandas as pd
import os
import gc

dirrs = []
for directory in dirrs:
    column_names = ["Volume", "Radius", "X","Y"]
    df = pd.DataFrame(columns = column_names)
    dfm = pd.DataFrame(columns = column_names)
    total_mask = np.zeros((0,0))
    
    for f1 in fnmatch.filter(os.listdir(directory),'*[0-9].tif'):
        print(f1)
        string = f1.split('_')
        col = int(string[-2])
        row = int(string[-1].split('.')[0])
        img = tifffile.imread(directory + '/' + f1)
        
        width = total_mask.shape[0]
        new_width = (col+1)*img.shape[0]
        height = total_mask.shape[1]
        new_height = (row+1)*img.shape[1]      
        total_mask = np.zeros((max(width,new_width),max(height,new_height)))
        
    for f2 in fnmatch.filter(os.listdir(directory),'*L.tif'):
        print(f2)
        string = f2.split('_')
        col = int(string[-3])
        row = int(string[-2])
        img = tifffile.imread(directory + '/' + f2)
        total_mask [(col)*img.shape[0]:(col+1)*img.shape[0],(row)*img.shape[1]:(row+1)*img.shape[1]  ]= img
    
    imgV = np.where(total_mask!=2, 0, total_mask)
    imgM = np.where(total_mask!=1, 0, total_mask)
    labeled_array, num_features = label(imgV)
    for v in range(1,num_features):
        print(str(v) + ' out of ' + str(num_features))
        vol = np.count_nonzero(labeled_array == v)
        rad = math.sqrt((vol / 1.25)/math.pi)
        thing = np.where(labeled_array == v)
        cx = thing[0].mean()
        cy = thing[1].mean()
        df2 = pd.DataFrame([[vol, rad, cx,cy]], columns=column_names)
        df = pd.concat([df, df2])
        
    
    from scipy import spatial
    theCenters = np.c_[df.X,df.Y]
    tree = spatial.KDTree(theCenters)
    df['friends_1000'] = tree.query_ball_point(theCenters, 875, workers = -1,return_length = True)
    
    labeled_arrayM, num_featuresM = label(imgM)
    for m in range(1,num_featuresM):
        print(str(m) + ' out of ' + str(num_featuresM))
        vol = np.count_nonzero(labeled_array == m)
        thing = np.where(labeled_arrayM == m)
        cx = thing[0].mean()
        cy = thing[1].mean()
        df2 = pd.DataFrame([[vol,0, cx,cy]], columns=column_names)
        dfm = pd.concat([dfm, df2])
    
    df.to_csv('ves_' + directory + '.csv', index=False)
    dfm.to_csv('mito_' + directory + '.csv', index=False)
    del total_mask
    del img
    del df
    del dfm
    del tree
    del imgV
    del imgM
    gc.collect()