"""
Thanks to Ian Lin
"""
import arff    # liac-arff
import numpy as np
import random
import xml.etree.ElementTree as ET
import pandas as pd
import ipdb

def load_data(dataset_path):
    arff_filename = dataset_path + ".arff"
    xml_filename = dataset_path + ".xml"
    X, Y = load_arff(arff_filename, xml_filename)
    return X, Y

def load_arff(arff_filename, xml_filename):
    # read arff file
    with open(arff_filename, "rb") as fp:
        data = arff.load(fp);

    # read xml file
    tree = ET.parse(xml_filename);
    root = tree.getroot()
    label_list = [];
    for child in root:
        label_list.append(child.attrib["name"]);
    column_list = [attr[0] for attr in data["attributes"]]
    feature_list = list(set(column_list) - set(label_list))

    # build converters to convert nominal data to numerical data
    converters = {}
    for attr in data["attributes"]:
        if attr[1] == 'NUMERIC':
            pass
        elif isinstance(attr[1], list):
            converter = {}
            for e, cls in enumerate(attr[1]):
                converter[cls] = e
            converters[attr[0]] = converter
        else:
            raise NotImplementedError("attribute {} is not supported.".format(att[1]))
    #print(converters, column_list, feature_list)

    # ipdb.set_trace()
    df = pd.DataFrame(data['data'], columns=column_list)
    df.replace(converters, inplace=True)
    # print "Read as sparse format";
    # n_instance = len(data["data"]);
    # dense_data = np.zeros( (n_instance, len(feature)+len(label)), dtype=float)
    # for i,instance in enumerate(data["data"]):
    #     for sf in instance:
    #         idx, val = sf.split(' ');
    #         dense_data[i][int(idx)] = val;
    # data = dense_data;

    X = df[feature_list].values;
    Y = df[label_list].values;
    if Y.dtype != np.int:
        raise ValueError("Y is not int.")

    return X, Y
