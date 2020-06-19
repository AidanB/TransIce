import tensorflow as tf
import numpy as np
import pickle

tt_dir = "D:/train_test/"
files = ["enc_en_training", "enc_en_test", "enc_is_training", "enc_is_test"]
arrs = {}
for filename in files:
    with open(tt_dir+filename,"rb") as file:
        temp = pickle.load(file)
        arr = np.array(temp)
        arrs[filename] = arr
