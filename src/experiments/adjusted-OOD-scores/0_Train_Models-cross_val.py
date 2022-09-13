#!/usr/bin/env python
# coding: utf-8

# In[36]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
sys.path.insert(0, "../")

from autogluon.vision import ImagePredictor, ImageDataset
import numpy as np
import pandas as pd
from utils.model_training import train_model, sum_xval_folds

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# ## Read data

# In[37]:


# path to data
CIFAR_10_DATA_PATH = "/datasets/uly/ood-data/cifar10_png/"
CIFAR_100_DATA_PATH = "/datasets/uly/ood-data/cifar100_png/"
MNIST_DATA_PATH = "/datasets/uly/ood-data/mnist_png/"
FASHION_MNIST_DATA_PATH = "/datasets/uly/ood-data/fashion_mnist_png/"

# read data from root folder
cifar_10_train_dataset, _, cifar_10_test_dataset = ImageDataset.from_folders(root=CIFAR_10_DATA_PATH)
cifar_100_train_dataset, _, cifar_100_test_dataset = ImageDataset.from_folders(root=CIFAR_100_DATA_PATH)
mnist_train_dataset, _, mnist_test_dataset = ImageDataset.from_folders(root=MNIST_DATA_PATH)
fashion_mnist_train_dataset, _, fashion_mnist_test_dataset = ImageDataset.from_folders(root=FASHION_MNIST_DATA_PATH)


# In[38]:


# dictionary to store data path and model

data_model_dict = {
    "cifar-10": {
        "train_data": cifar_10_train_dataset,
        "test_data": cifar_10_test_dataset,
    },
    "cifar-100": {
        "train_data": cifar_100_train_dataset,
        "test_data": cifar_100_test_dataset,
    },
    "mnist": {
        "train_data": mnist_train_dataset,
        "test_data": mnist_test_dataset,
    },
    "fashion-mnist": {
        "train_data": fashion_mnist_train_dataset,
        "test_data": fashion_mnist_test_dataset,
    },
}


# In[39]:


# Create mini train dataset for testing
def get_imbalanced_dataset(dataset, fractions):
    assert len(fractions) == dataset['label'].nunique()

    imbalanced_dataset = pd.DataFrame(columns=dataset.columns)
    print(imbalanced_dataset)
    for i in range(len(fractions)):
        idf = dataset[dataset['label'] == i].sample(frac=fractions[i])
        print(f'label {i} will have {idf.shape[0]} examples')
        imbalanced_dataset = pd.concat([imbalanced_dataset, idf], ignore_index=True)
    print(f'total imbalanced dataset length {imbalanced_dataset.shape[0]}')
    return imbalanced_dataset

### Uncomment below to create imbalanced datasets

# cifar_100_num_classes = len(cifar_100_train_dataset['label'].unique())
# cifar_100_distribution = [0.15] * int(cifar_100_num_classes * 0.9) + [1.] * int(cifar_100_num_classes * 0.1)
# cifar_100_train_dataset = get_imbalanced_dataset(cifar_100_train_dataset, cifar_100_distribution)
# cifar_10_train_dataset = get_imbalanced_dataset(cifar_10_train_dataset,[0.09,0.09,0.09,0.09,1.,1.,0.09,0.09,1.,1.])
# mnist_train_dataset = get_imbalanced_dataset(mnist_train_dataset,[0.09,0.09,0.09,0.09,1.,1.,0.09,0.09,1.,1.])
# fashion_mnist_train_dataset = get_imbalanced_dataset(fashion_mnist_train_dataset,[0.09,0.09,0.09,0.09,1.,1.,0.09,0.09,1.,1.])


# In[40]:


# Check out a dataset
mnist_train_dataset.head()


# ## Train model

# In[41]:


get_ipython().system('mkdir models # Create models folder to save model results into')


# In[42]:


# # load pickle file util
# import pickle

# def _load_pickle(pickle_file_name, verbose=1):
#     """Load pickle file"""
#     if verbose:
#         print(f"Loading {pickle_file_name}")
#     with open(pickle_file_name, 'rb') as handle:
#         out = pickle.load(handle)
#     return out

# def sum_xval_folds(model, model_results_folder, num_cv_folds=5, verbose=1, **kwargs):
#     # get original label name to idx mapping
#     label_name_to_idx_map = {'airplane': 0,
#                          'automobile': 1,
#                          'bird': 2,
#                          'cat': 3,
#                          'deer': 4,
#                          'dog': 5,
#                          'frog': 6,
#                          'horse': 7,
#                          'ship': 8,
#                          'truck': 9}
#     results_list = []
    
#     # get shapes of arrays (this is dumb way to do it what is better?)
#     pred_probs_shape = []
#     features_shape = []
#     labels_shape = []
#     for split_num in range(num_cv_folds):

#         out_subfolder = f"{model_results_folder}_{model}/split_{split_num}/"

#         # pickle file name to read
#         get_pickle_file_name = (
#             lambda object_name: f"{out_subfolder}_{object_name}_split_{split_num}"
#         )

#         # NOTE: the "test_" prefix in the pickle name correspond to the "test" split during cross-validation.
#         print(_load_pickle(get_pickle_file_name("test_pred_probs")))
#         pred_probs_split = _load_pickle(get_pickle_file_name("test_pred_probs"), verbose=verbose)
#         labels_split = _load_pickle(get_pickle_file_name("test_labels"), verbose=verbose)
#         test_pred_features_split = _load_pickle(get_pickle_file_name("test_pred_features"), verbose=verbose)

#         pred_probs_shape.append(pred_probs_split)
#         features_shape.append(test_pred_features_split)
#         labels_shape.append(labels_split)
#     print('41 done')

#     pred_probs_shape = np.vstack(pred_probs_shape)
#     labels_shape = np.hstack(labels_shape)
        
#     pred_probs = np.zeros_like(pred_probs_shape)
#     labels = np.zeros_like(labels_shape)
#     images = np.empty((labels_shape.shape[0],) ,dtype=object)
    
#     print(pred_probs.shape, labels.shape, images.shape)

#     for split_num in range(num_cv_folds):

#         out_subfolder = f"{model_results_folder}_{model}/split_{split_num}/"
#         print(out_subfolder)

#         # pickle file name to read
#         get_pickle_file_name = (
#             lambda object_name: f"{out_subfolder}_{object_name}_split_{split_num}"
#         )
        
#         print(get_pickle_file_name("test_pred_probs"))

#         # NOTE: the "test_" prefix in the pickle name correspond to the "test" split during cross-validation.
#         pred_probs_split = _load_pickle(get_pickle_file_name("test_pred_probs"), verbose=verbose)
#         labels_split = _load_pickle(get_pickle_file_name("test_labels"), verbose=verbose)
#         images_split = _load_pickle(get_pickle_file_name("test_image_files"), verbose=verbose)
#         indices_split = _load_pickle(get_pickle_file_name("test_indices"), verbose=verbose)
#         indices_split = np.array(indices_split)
        
#         print(get_pickle_file_name("test_indices"))
        
#         pred_probs[indices_split] = pred_probs_split
#         labels[indices_split] = labels_split
#         images[indices_split] = np.array(images_split)
#         print('42 done')
#     print('43')
#     return pred_probs, labels, images


# In[51]:


get_ipython().run_cell_magic('time', '', '\ndef train_ag_model(\n    train_data,\n    dataset_name,\n    model_folder="./models/",    \n    epochs=100,\n    model="swin_base_patch4_window7_224",\n    time_limit=10*3600\n):\n    \n    train_args = {  \n        "num_cv_folds": 5, \n        "epochs": 100, \n        "time_limit": 10*3600, \n        "random_state": 123,\n        "verbose": 0, \n    }\n\n    model_results_folder = f"{dataset_name}"\n    \n    # Train model\n    clf = train_model(model, train_data, model_results_folder, **train_args);\n    train_pred_probs, labels, images = sum_xval_folds(model, model_results_folder, **train_args)\n    \n    # save model\n    filename = f"{model_results_folder}{model}_{dataset_name}.ag"\n    clf.save(filename)\n    \n    return clf\n')


# ## Train model for all datasets

# In[ ]:


model = "swin_base_patch4_window7_224"

for key, data in data_model_dict.items():

    dataset = key
    train_dataset = data["train_data"]
    
    print(f"Dataset: {dataset}")
    print(f"  Records: {train_dataset.shape}")
    print(f"  Classes: {train_dataset.label.nunique()}")    
    
    _ = train_ag_model(train_dataset, dataset_name=dataset, model=model, epochs=100)


# In[ ]:


# run before push because sumxvalfolds was broken
model_results_folder = "cifar-100"
model = "swin_base_patch4_window7_224"
train_pred_probs, labels, images = sum_xval_folds(model, model_results_folder, **train_args)

