"""
Thanks to Ian Lin for data loading functions
"""
import numpy as np
import gzip
import pickle
import itertools as itt
import os
import arff    # liac-arff
import xml.etree.ElementTree as ET
import pandas as pd

def load_data(dataset_path: str):
    """Dataset loading function for dataset downloaded from mulan.
    """
    arff_filename = dataset_path + ".arff"
    xml_filename = dataset_path + ".xml"
    X, Y = load_arff(arff_filename, xml_filename)
    return X, Y

def load_arff(arff_filename: str, xml_filename: str):
    # read arff file
    with open(arff_filename, "r") as fp:
        data = arff.load(fp)

    # read xml file
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    label_list = []
    for child in root:
        label_list.append(child.attrib["name"])
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
    # print "Read as sparse format"
    # n_instance = len(data["data"])
    # dense_data = np.zeros( (n_instance, len(feature)+len(label)), dtype=float)
    # for i,instance in enumerate(data["data"]):
    #     for sf in instance:
    #         idx, val = sf.split(' ')
    #         dense_data[i][int(idx)] = val
    # data = dense_data

    X = df[feature_list].values
    Y = df[label_list].values
    if Y.dtype != np.int:
        raise ValueError("Y is not int.")

    return X, Y

def pairwise_hamming(Z, Y):
    """
    Z and Y should be the same size 2-d matrix
    """
    return -np.abs(Z - Y).mean(axis=1)


def pairwise_f1(Z, Y):
    """
    Z and Y should be the same size 2-d matrix
    """
    # calculate F1 by sum(2*y_i*h_i) / (sum(y_i) + sum(h_i))
    Z = Z.astype(int)
    Y = Y.astype(int)
    up = 2*np.sum(Z & Y, axis=1).astype(float)
    down1 = np.sum(Z, axis=1)
    down2 = np.sum(Y, axis=1)

    down = (down1 + down2)
    down[down==0] = 1.
    up[down==0] = 1.

    #return up / (down1 + down2)
    #assert np.all(up / (down1 + down2) == up/down) == True
    return up / down


def pairwise_rankloss(Z, Y): #truth(Z), prediction(Y)
    """
    Z and Y should be the same size 2-d matrix
    """
    rankloss = ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie0 = 0.5 * ((Z==0) & (Y==0)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie1 = 0.5 * ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==1)).sum(axis=1)
    return -(rankloss + tie0 + tie1)


def pairwise_acc(Z, Y):
    f1 = 1.0 * ((Z>0) & (Y>0)).sum(axis=1)
    f2 = 1.0 * ((Z>0) | (Y>0)).sum(axis=1)
    f1[f2<=0] = 1.0
    f1[f2>0] /= f2[f2>0]
    return f1


def get_scoring_fn(scoring):
    if scoring == 'hamming':
        scoring_fn = pairwise_hamming
    elif scoring == 'f1':
        scoring_fn = pairwise_f1
    elif scoring == 'rankloss':
        scoring_fn = pairwise_rankloss
    elif scoring == 'acc':
        scoring_fn = pairwise_acc
    else:
        print("err:", [scoring])
    return scoring_fn

def seed_random_state(seed):
    """Turn seed into np.random.RandomState instance
    """
    if (seed is None) or (isinstance(seed, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r can not be used to generate numpy.random.RandomState"
                     " instance" % seed)
