import tensorflow as tf
import numpy as np
import pickle

tt_dir = "D:/train_test/"
files = ["enc_en_training", "enc_en_test", "enc_is_training", "enc_is_test"]

def get_encoded(verbose=False):
    arrs = {}
    for filename in files:
        if verbose:
            print("Loading encoded file {}".format(filename))
        with open(tt_dir+filename,"rb") as file:
            temp = pickle.load(file)
            arr = np.array(temp)
            arrs[filename] = arr

    return arrs

def get_angles(pos,i,d_model):
    print