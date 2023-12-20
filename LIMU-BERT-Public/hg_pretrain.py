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
from models import LIMUBertModel4Pretrain, MultiMAE
from utils import set_seeds, get_device, LIBERTMultiDataset4Pretrain, handle_argv, load_pretrain_data_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, Preprocess4Normalization,  Preprocess4Mask


def preprocess_one_csv(path, seq_len):
    df = pd.read_csv(path)
    df = df.dropna()
    df['RightGazeDirection'] = df['RightGazeDirection'].apply(lambda x: ast.literal_eval(x))
    df['Unit_Vector'] = df['Unit_Vector'].apply(lambda x: ast.literal_eval(x))
    gaze = df["RightGazeDirection"].values.tolist()
    head = df["Unit_Vector"].values.tolist()
    if not len(gaze) < seq_len:
        gaze = np.array(gaze[:(len(gaze)//seq_len * seq_len)])
        gaze = np.array(np.split(gaze, len(gaze)//seq_len))
        head = np.array(head[:(len(head)//seq_len * seq_len)])
        head = np.array(np.split(head, len(head)//seq_len))
        return gaze, head


def preprocess_hgbd_dataset(args):
    dataset_cfg = args.dataset_cfg
    gaze = np.empty((0, dataset_cfg.seq_len, dataset_cfg.dimension))
    head = np.empty((0, dataset_cfg.seq_len, dataset_cfg.dimension))
    for per in os.listdir("../Data/Version2"):
        print(per)
        try:
            csvs = os.listdir("../Data/Version2/" + per)
            for f in range(len(csvs)):
                temp1, temp2 = preprocess_one_csv("../Data/Version2/" + per + "/" + csvs[f], dataset_cfg.seq_len)
                if temp1 is not None and temp2 is not None:
                    gaze = np.concatenate((gaze, temp1), axis=0)
                    head = np.concatenate((head, temp2), axis=0)
        except:
            pass
    assert head.shape == gaze.shape
    np.save("../LIMU-BERT-Public/dataset/hgbd/data_2.npy", gaze)
    np.save("../LIMU-BERT-Public/dataset/hgbd/label_2.npy", head)
    # np.save("../LIMU-BERT-Public/dataset/hgbd/label_2.npy", np.ones(gaze.shape))


def main(args, training_rate):
    preprocess_hgbd_dataset(args)
    gdata, hdata, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    # pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    pipeline = [Preprocess4Mask(mask_cfg)]
    gdata_train, hdata_train, gdata_test, hdata_test = prepare_pretrain_dataset(gdata, hdata, training_rate, seed=train_cfg.seed)

    data_set_train = LIBERTMultiDataset4Pretrain(gdata_train, hdata_train, pipeline=pipeline)
    data_set_test = LIBERTMultiDataset4Pretrain(gdata_test, hdata_test, pipeline=pipeline)
    
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    
    model = MultiMAE(model_cfg)    
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

    def func_loss(model, batch):
        gmask_seqs, gmasked_pos, gseqs, hmask_seqs, hmasked_pos, hseqs = batch
        gseq_recon = model(gmask_seqs, hmask_seqs, gmasked_pos)
        #import pdb; pdb.set_trace()
        gloss_lm = criterion(gseq_recon, gseqs) # for masked LM
        return gloss_lm

    def func_forward(model, batch):
        gmask_seqs, gmasked_pos, gseqs, hmask_seqs, hmasked_pos, hseqs = batch
        gseq_recon = model(gmask_seqs, hmask_seqs, gmasked_pos)
        return gseq_recon, gseqs

    def func_evaluate(seqs, gpredict_seqs):
        gloss_lm = criterion(gpredict_seqs, seqs)
        return gloss_lm.mean().cpu().numpy()

    if hasattr(args, 'pretrain_model'):
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test
                      , model_file=args.pretrain_model)
    else:
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=None)


if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    main(args, training_rate)
