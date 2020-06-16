from glob import glob
from collections import defaultdict
import re

extracted_dir = "D:/processed/"
saved_list_handler1 = re.compile(r"(?:\('[0-9]+', \[)(.+)(?:\]\))")

def get_vocab(verbose=False):

    en_filelist = glob(extracted_dir+"en_*")
    is_filelist = glob(extracted_dir+"is_*")

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


def process_saved_list(inLine):
    just_tokens_from_list = saved_list_handler1.findall(inLine)[0]
    split_tokens = just_tokens_from_list.split(", ")

    clean_tokens = [x[1:-1].lower() for x in split_tokens]

    return clean_tokens

get_vocab(verbose=True)