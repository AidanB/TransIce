from glob import glob
from collections import defaultdict
import re
from random import shuffle
from os.path import basename
import pickle

extracted_dir = "D:/processed/"
target_dir = "D:/train_test/"
saved_list_handler1 = re.compile(r"(?:\('[0-9]+ .+?\.xml', \[)(.+)(?:\]\))") # extracts tokens
saved_list_handler2 = re.compile(r"(?:\(')([0-9]+ .+?\.xml)") # extracts sent id number & doc title

def create_train_split(split=0.9,verbose=False):
    en_filelist = glob(extracted_dir + "en_*")
    is_filelist = glob(extracted_dir + "is_*")

    paired = defaultdict(lambda: [])

    def handle_files(filelist):
        for filename in filelist:
            with open(filename,"r",encoding="utf8") as file:
                lines = [line.strip() for line in file.readlines()]
                if verbose:
                    print("{0}: {1}".format(filename, len(lines)))
                for line in lines:
                    try:
                        key = saved_list_handler2.findall(line)[0]
                    except IndexError:
                        print(line)
                    paired[key].append(line)

    handle_files(en_filelist)
    handle_files(is_filelist)

    split_num = int(split * len(paired))
    print(split_num) # debug purposes

    paired = list(paired.values())
    shuffle(paired)

    training_data = paired[:split_num]
    test_data = paired[split_num:]

    def save_files():
        train_file_en = open(target_dir+"en_training","w+",encoding="utf8")
        train_file_is = open(target_dir+"is_training","w+",encoding="utf8")
        test_file_en = open(target_dir+"en_test","w+",encoding="utf8")
        test_file_is = open(target_dir+"is_test","w+",encoding="utf8")

        for line in training_data: # enumerated for debugging purposes, remember to remove
            if not saved_list_handler2.findall(line[0])[0] == saved_list_handler2.findall(line[1])[0]:
                print(line)
            if not len(line) == 2:
                print(line)
            print(line[0],file=train_file_en)
            print(line[1],file=train_file_is)


        train_file_en.flush()
        train_file_en.close()
        train_file_is.flush()
        train_file_is.close()

        for line in test_data:
            print(line[0],file=test_file_en)
            print(line[1],file=test_file_is)

        test_file_en.close()
        test_file_is.close()

    save_files()



def process_saved_list(inLine):
    just_tokens_from_list = saved_list_handler1.findall(inLine)[0]
    split_tokens = just_tokens_from_list.split(", ")

    clean_tokens = [x[1:-1].lower() for x in split_tokens]

    return clean_tokens

def get_vocab(unk_thresh=2,verbose=False):

    en_train = target_dir+"en_training"
    is_train = target_dir+"is_training"

    def extractor(filename):
        vocab = set()
        count_vocab = defaultdict(lambda: 0)
        if verbose:
            print("Processing file: {}".format(filename))
        with open(filename,"r",encoding="utf8") as file:
            lines = file.readlines()
            for line in lines:
                processed = process_saved_list(line)
                for token in processed:
                    vocab.add(token)
                    count_vocab[token] += 1

        counter = 2 # 0 for pad, 1 for unk
        enc_vocab = {}
        for word in vocab:
            if count_vocab[word] < unk_thresh:
                enc_vocab[word] = 1
            else:
                enc_vocab[word] = counter
                counter += 1

        return enc_vocab

    en_vocab = extractor(en_train)
    is_vocab = extractor(is_train)

    print(en_vocab) # debug
    print(is_vocab) # debug

    return en_vocab, is_vocab

def encode(vocabs, unk_thresh=2,max_length=40):
    files = [target_dir + "en_training", target_dir + "en_test", target_dir + "is_training", target_dir + "is_test"]
    langs = [0,0,1,1]

    def enc_file(filename,lang):
        all_enc = []
        with open(filename,"r",encoding="utf8") as file:
            for line in file:
                enc = []
                tokens = process_saved_list(line)
                for token in tokens:
                    if token in vocabs[lang]:
                        enc.append(vocabs[lang][token])
                    else:
                        enc.append(1)

                if len(enc) < max_length:
                    enc = [0 for x in range(max_length - len(enc))] + enc
                elif len(enc) > max_length:
                    enc = enc[0:max_length]

                all_enc.append(enc)

        return all_enc

    encoded_files = []
    for i,filename in enumerate(files):
        encoded_files.append(enc_file(filename,langs[i]))

    for i,enc in enumerate(encoded_files):
        filename = "enc_"+basename(files[i])

        with open(target_dir+filename,"wb+") as file:
            pickle.dump(enc,file)

    vocab_names = ["en_vocab","is_vocab"]
    for i,name in enumerate(vocab_names):
        with open(target_dir+name,"wb+") as vocab_file:
            pickle.dump(vocabs[i],vocab_file)



#create_train_split(verbose=True)

en_vocab, is_vocab = get_vocab(verbose=True)
encode([en_vocab,is_vocab])