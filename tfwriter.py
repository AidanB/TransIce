from tensorflow.train import *
from tensorflow.io import TFRecordWriter
import pickle

tt_dir = "D:/train_test/"

#list_filenames = ["enc_en_training","enc_en_test","enc_is_training","enc_is_test"]
list_filenames = ["enc_en_training200000-50000","enc_en_test200000-50000","enc_is_training200000-50000","enc_is_test200000-50000"]

def load_list_file(filename,verbose=True):
    if verbose:
        print("Loading file {0}...".format(filename))
    with open(tt_dir+filename,"rb") as list_file:
        loaded_list = pickle.load(list_file)

    return loaded_list

def numlist_to_str(inList):
    result_str = ""
    for num in inList:
        result_str += str(num) + "_"

    return result_str

def create_FeatureLists(inList):
    feat_step1 = [Feature(int64_list=Int64List(value=x)) for x in inList]
    #feat_step2 = [FeatureList(feature=x) for x in feat_step1]
    feat_step2 = []
    for feat in feat_step1:
        temp = [feat]
        feat_list = FeatureList(feature=temp)
        feat_step2.append(feat_list)
    feat_lists = [FeatureLists(feature_list={"features": x}) for x in feat_step2]
    seq_exs = [SequenceExample(feature_lists=x) for x in feat_lists]

    return seq_exs

list_en_train = load_list_file(list_filenames[0])
list_en_test = load_list_file(list_filenames[1])
list_is_train = load_list_file(list_filenames[2])
list_is_test = load_list_file(list_filenames[3])

fl_en_train = create_FeatureLists(list_en_train)
fl_en_test = create_FeatureLists(list_en_test)
fl_is_train = create_FeatureLists(list_is_train)
fl_is_test = create_FeatureLists(list_is_test)

#towrite = {"en_train":SequenceExample(feature_lists=fl_en_train),"en_test":SequenceExample(feature_lists=fl_en_test),"is_train":SequenceExample(feature_lists=fl_is_train),"is_test":SequenceExample(feature_lists=fl_is_test),}
towrite = {"en_train":fl_en_train,"en_test":fl_en_test,"is_train":fl_is_train,"is_test":fl_is_test}
#print(towrite_en_train)

for name in towrite:
    ses = towrite[name]
    filename = name + ".tfrecord"
    with TFRecordWriter(filename) as writer:
        for ex in ses:
            writer.write(ex.SerializeToString())