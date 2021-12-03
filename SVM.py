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

print(f'[Phase 1 completed in {round(time.time() - startTime, 2)} seconds]')



startTime = time.time()

# ------------------------------------------------------------------------------------------------
# Phase 2: Building/Training the model
# ------------------------------------------------------------------------------------------------

'''
## ATTEMPT 1
## Here was my initial attempt. I was using GridSearchCV to try and find the best hyperparameters.

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

svc = SVC(probability=True)

# construct our model using the classifier and different parameter values. When we call fit(),
# the model will try the parameters to find the best ones. I specify the n_jobs parameter to 
# attempt to speed up fit() process via parallelization. My computer crashed when I
# parallelized across all cpu cores, so I'm doing two less than the max, for a total of ten.
model = GridSearchCV(svc, param_grid)

model.fit(x_train, y_train)
'''

# ------------------------------------------------------------------------------------------------

'''
## ATTEMPT 2
## Attempt 1 was taking a scarily long amount of time. I dug in and realized that with the 
## parameter grid configured the way it was in atttempt 1, it was running the model.fit() method
## 160 times - there are 32 different hyperparameter combinations and with GridSearchCV's
## default of 5-fold cross validation, it was running 32 * 5 = 160 times. So in this attempt, I
## changed the param_grid to only have one combination of parameters so that it would only run 
## 1 * 5 = 5 times.

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {  
    'C': [1],
    'gamma': [1],
    'kernel': ['rbf']
}

svc = SVC(probability=True)
model = GridSearchCV(svc, param_grid)

model.fit(x_train, y_train)
'''

# ------------------------------------------------------------------------------------------------

'''
## ATTEMPT 3
## I realized that I could probably speed up the training if I took advantage of parallelization,
## so I specified the n_jobs parameter to take advantage of all my cpu cores. My computer had a
## hard crash with error "kernel panic" when doing that, so I set n_jobs equal to two less than
## the number of cpu cores in my computer. My computer crashed again, so I went into my computer
## and cleared out some storage in case it was out of RAM and was also out of swap memory. I 
## monitored the activity monitor to see how much RAM/CPU was being used, and it was essentially
## maxed out.

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {  
    'C': [1],
    'gamma': [1],
    'kernel': ['rbf']
}

svc = SVC(probability=True)
model = GridSearchCV(svc, param_grid, n_jobs=os.cpu_count()-2, verbose=5)

model.fit(x_train, y_train)
'''

# ------------------------------------------------------------------------------------------------

'''
## ATTEMPT 4
## After failing to take advantage of parallelization, I thought I should try reducing the calls
## to model.fit() once again. To do this, I set the cv parameter to only perform two-fold
## cross-validation resulting in two calls to fit().

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {  
    'C': [1],
    'gamma': [1],
    'kernel': ['rbf']
}

svc = SVC(probability=True)
model = GridSearchCV(svc, param_grid, cv=2, verbose=5)

model.fit(x_train, y_train)
'''

# ------------------------------------------------------------------------------------------------

'''
## ATTEMPT 5
## Attempt 4 still took absolutely forever and never terminated, so I dropped the use of 
## GridSearchCV altogether, resulting in this attempt's code.

from sklearn.svm import SVC

svc = SVC(kernel='rbf', probability=True)

svc.fit(x_train, y_train)
'''

# ------------------------------------------------------------------------------------------------

'''
## ATTEMPT 6
## The pattern continues... attempt 5 still never terminated, so I increased the cache size to
## see if I could get any speedup from that, and I set the max iterations to a ridiculously low
## number just as a proof-of-concept to see if I could get the process to terminate, which it
## never did.

from sklearn.svm import SVC

svc = SVC(kernel='rbf', probability=True, cache_size=1900, verbose=True, max_iter=5)

svc.fit(x_train, y_train)
'''

# ------------------------------------------------------------------------------------------------

## ATTEMPT 7
## Here is my last attempt, which was to use LinearSVC instead of SVC. Per sklearn.svm.SVC's
## docstring, "The fit time scales at least quadratically with the number of samples and may be
## impractical beyond tens of thousands of samples. For large datasets consider using ~sklearn.svm.LinearSVC"
## As such, I am trying LinearSVC here on our decently large dataset. I set the tolerance to a
## rather large number and kept the max_iter set to a very low number, again as a 
## proof-of-concept to see if it would terminate/converge. The process did terminate, with a 
## message of "ConvergenceWarning: Liblinear failed to converge, increase the number of
## iterations." Another group member, Steven, tried the code with max_iter=100 and still got 
## the same ConvergenceWarning. This may be because the data is simply not linearly-seperable.

from sklearn.svm import LinearSVC

# construct our classifier
svc = LinearSVC(tol=0.01, verbose=5, max_iter=5)

svc.fit(x_train, y_train)


print(f'[Phase 2 completed in {round(time.time() - startTime, 2)} seconds]')