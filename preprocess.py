# needs to deal with the errant xml file in the directory, for cleanliness' sake

import xml.etree.ElementTree as ET
import re
from glob import glob
from os import listdir
from os.path import basename

# top level dir for xml files
top_dir = "D:/ParIce1.1/xml/"
dest_dir = "D:/processed/"

# list of subdirs
subdirs = listdir(top_dir)

def extract_from_dir(dir):
    en_files = glob(top_dir+dir+"/en/"+"*.xml")
    is_files = glob(top_dir+dir+"/is/"+"*.xml")

    # align filenames
    en_files = sorted(en_files)
    is_files = sorted(is_files)

    all_en_sents = []
    all_is_sents = []

    for en_file in en_files:
        all_en_sents.extend(extract_from_file(en_file))
    for is_file in is_files:
        all_is_sents.extend(extract_from_file(is_file))

    return all_en_sents, all_is_sents


def extract_from_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    namespace = re.findall(r"{.+}",root.tag)[0]
    xml_sents = root.iter(namespace+"seg")

    all_sents = []
    for sent in xml_sents:
        top_of_sent = sent[0]
        i = sent.attrib["id"] # extract index from tag. should be equivalent to iteration index but better safe than sorry

        tokens = []

        for token in top_of_sent:
            tokens.append(token.text)

        all_sents.append((i,tokens))

    return all_sents

def extract_all(verbose=False):
    for dir in subdirs:
        if verbose:
            print("Processing file {}".format(dir))
        temp_en,temp_is = extract_from_dir(dir)
        with open(dest_dir+"en_"+basename(dir),"w+",encoding="utf8") as en_targ:
            for sent in temp_en:
                print(sent,file=en_targ)
        with open(dest_dir+"is_"+basename(dir),"w+",encoding="utf8") as is_targ:
            for sent in temp_is:
                print(sent,file=is_targ)

extract_all(verbose=True)

