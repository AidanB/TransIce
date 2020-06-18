from glob import glob
from collections import defaultdict
import re
from random import shuffle

extracted_dir = "D:/processed/"
target_dir = "D:/train_test/"
saved_list_handler1 = re.compile(r"(?:\('[0-9]+', \[)(.+)(?:\]\))") # extracts tokens
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





def get_vocab(verbose=False):

    en_filelist = glob(extracted_dir+"en_*")
    is_filelist = glob(extracted_dir+"is_*")

    def process_saved_list(inLine):
        just_tokens_from_list = saved_list_handler1.findall(inLine)[0]
        split_tokens = just_tokens_from_list.split(", ")

        clean_tokens = [x[1:-1].lower() for x in split_tokens]

        return clean_tokens

    def extractor(filelist):
        vocab = set()
        count_vocab = defaultdict(lambda: 0)
        for filename in filelist:
            if verbose:
                print("Processing file: {}".format(filename))
            with open(filename,"r",encoding="utf8") as file:
                lines = file.readlines()
                for line in lines:
                    processed = process_saved_list(line)
                    for token in processed:
                        vocab.add(token)
                        count_vocab[token] += 1

        return vocab, count_vocab

    en_vocab, en_count = extractor(en_filelist)
    is_vocab, is_count = extractor(is_filelist)



create_train_split(verbose=True)

#get_vocab(verbose=True)