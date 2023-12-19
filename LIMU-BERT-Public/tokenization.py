#!/usr/bin/env python
import argparse
import sys
import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
import copy
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader

import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import LIMUBertModel4Pretrain, Embeddings
from utils import set_seeds, get_device, LIBERTDataset4Pretrain, handle_argv, load_pretrain_data_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, Preprocess4Normalization,  Preprocess4Mask


def preprocess_one_csv(path, seq_len):
    df = pd.read_csv(path)
    df = df.dropna()
    df['RightGazeDirection'] = df['RightGazeDirection'].apply(lambda x: ast.literal_eval(x))
    gaze = df["RightGazeDirection"].values.tolist()
    if not len(gaze) < seq_len:
        gaze = np.array(gaze[:(len(gaze)//seq_len * seq_len)])
        gaze = np.array(np.split(gaze, len(gaze)//seq_len))
        return gaze


def preprocess_hgbd_dataset():
    seq_len=240
    gaze = np.empty((0, 240, 3))
    for per in os.listdir("../Data/Version2"):
        print(per)
        try:
            csvs = os.listdir("../Data/Version2/" + per)
            for f in range(len(csvs)):
                temp = preprocess_one_csv("../Data/Version2/" + per + "/" + csvs[f], seq_len)
                if temp is not None:
                    gaze = np.concatenate((gaze, temp), axis=0)
        except:
            pass
    print(gaze.shape)
    np.save("../LIMU-BERT-Public/dataset/hgbd/data_2.npy", gaze)
    np.save("../LIMU-BERT-Public/dataset/hgbd/label_2.npy", np.ones(gaze.shape))


def main(args, training_rate):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    # pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    pipeline = [Preprocess4Mask(mask_cfg)]
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)

    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    #data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    #data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    
    model = Embeddings(model_cfg)
    for i, batch in enumerate(data_loader_train):
        mask_seqs, masked_pos, seqs = batch
        print(mask_seqs.shape)
        tokens = model(mask_seqs)
        print(tokens.shape)
        break              
        

if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    #preprocess_hgbd_dataset()
    main(args, training_rate)
