import xml.etree.ElementTree as ET
import re
from glob import glob
from os import listdir
from os.path import isdir, isfile

# top level dir for xml files
top_dir = "D:/ParIce1.1/xml/"

# list of subdirs
subdirs = listdir(top_dir)

def extract_from_dir(dir):
    en_files = glob(top_dir+dir+"/en/"+"*.xml")
    is_files = glob(top_dir+dir+"/is/"+"*.xml")

    # align filenames
    en_files = sorted(en_files)
    is_files = sorted(is_files)

    a = extract_from_file(en_files[0])
    for sent in a:
        print(sent)

def extract_from_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    namespace = re.findall(r"{.+}",root.tag)[0]
    xml_sents = root.iter(namespace+"seg")

    all_sents = []
    for i,sent in enumerate(xml_sents):
        top_of_sent = sent[0]

        tokens = []

        for token in top_of_sent:
            tokens.append(token.text)

        all_sents.append((i,tokens))

    """
    text = root.iter(namespace+"text")

    for child in text:
        for subchild in child:
            print(subchild)

    #xml_sents = tree.iter("seg")
    """
    return all_sents

extract_from_dir(subdirs[0])

