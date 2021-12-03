"""
CIS 4930 Project File: Support Vector Machine Classification

This script closely follows the write-up by Vegi Shanmukh at 
https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01
yet was configured to match the data from https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?select=Train

"""

import time

startTime = time.time()

# ------------------------------------------------------------------------------------------------
# Phase 1: Preparing our data
# ------------------------------------------------------------------------------------------------

import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pandas as pd


flat_data_arr: list[np.ndarray] = [] # input array
target_arr: list[int] = [] # output array

# categories=['Cars','Ice cream cone','Cricket ball']
streetsigns = [f'{n}' for n in range(43)]
datadir = 'StreetSigns/Train'

for sign in streetsigns:
    categoryFolder = os.path.join(datadir, sign)
    
    for img in os.listdir(categoryFolder):
        ## read individual image into an MxNx3 array, for RGB values
        img_array: np.ndarray = imread( os.path.join(categoryFolder, img) )
        
        ## resize each image to same dimensions, which is necessary for the SVM
        img_resized: np.ndarray = resize(img_array, (150,150,3)) # 150*150*3 = 67_500 features
        
        ##  store each image's data and category
        flat_data_arr.append( img_resized.flatten() )   # insert the flattened image as an ndarray into the list
        # The following line is slightly redundant in our case, but makes more
        # sense when the category names are strings of characters, not numbers.
        target_arr.append( streetsigns.index(sign) ) # insert the category name (i.e., a num on [0,42])


## convert Python lists into NumPy arrays
# `flat_data` and `target` have a len of (147 images per sign) * (43 signs) = 6321 rows.
# each row (i.e., each list item) is an ndarray of len 67_500
flat_data = np.array(flat_data_arr)
target    = np.array(target_arr)

df = pd.DataFrame(flat_data)
df['Target'] = target
x_train = df.iloc[:,:-1] # input data (feature vectors) (all except last column)
y_train = df.iloc[:,-1]  # output data (class labels with key 'Target') (only last column)

#
# Phase 1 code (input data configuration) takes around 27 seconds to run.
print(f'[Phase 1 completed in {round(time.time() - startTime, 2)} seconds]')
#



startTime = time.time()

# ------------------------------------------------------------------------------------------------
# Phase 2: Building the model
# ------------------------------------------------------------------------------------------------

from sklearn.svm import SVC, LinearSVC

# from sklearn.model_selection import GridSearchCV

# construct our classifier
# svc = SVC(kernel='rbf', probability=True, cache_size=1900, verbose=True, max_iter=5)

svc = LinearSVC(tol=0.01, verbose=5, max_iter=5)

# param_grid = {  
#     'C': [0.1, 1, 10, 100],
#     'gamma': [0.0001, 0.001, 0.1, 1],
#     'kernel': ['rbf', 'poly']
# }

svc.fit(x_train, y_train)

# param_grid = {  
#     'C': [1],
#     'gamma': [1],
#     'kernel': ['rbf']
# }

# construct our model using the classifier and different parameter values. When we call fit(),
# the model will try the parameters to find the best ones. I specify the n_jobs parameter to 
# attempt to speed up fit() process via parallelization. My computer crashed when I
# parallelized across all cpu cores, so I'm doing two less than the max, for a total of ten.
# model = GridSearchCV(svc, param_grid, n_jobs=os.cpu_count()-2, verbose=5)
# model = GridSearchCV(svc, param_grid, cv=2, verbose=5)

#
# Phase 2 code (model construction) takes around 6.5 seconds to run.
print(f'[Phase 2 completed in {round(time.time() - startTime, 2)} seconds]')
#



# startTime = time.time()

# # model.fit(x_train, y_train)

# print(f'[Phase 3 completed in {round(time.time() - startTime, 2)} seconds]')


# ------------------------------------------------------------------------------------------------
# Phase 3: Train the model
# ------------------------------------------------------------------------------------------------

# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

# print(type(x_train), type(x_test), type(y_train), type(y_test))

# print('Splitted Successfully')

# model.fit(x_train, y_train)

# print('The Model is trained well with the given images')
# model.best_params_ contains the best parameters obtained from GridSearchCV