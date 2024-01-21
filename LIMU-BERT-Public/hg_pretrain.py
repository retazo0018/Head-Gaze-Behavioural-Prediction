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
from datetime import datetime
import models, train, tracking, plot
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import LIMUBertModel4Pretrain, LIMUBertMultiMAEModel4Pretrain, LIMUBertAEModel4Pretrain
from utils import set_seeds, get_device, LIBERTMultiDataset4Pretrain,LIBERTGazeDataset4Pretrain, handle_argv, load_pretrain_data_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, Preprocess4Normalization,  Preprocess4Mask
import mlflow
from statistic import compute_dtw_metric, compute_levenschtein_distance
from hyperparameter_opt import HyperparameterOptimization
import itertools


def preprocess_one_csv(path, seq_len, downsample_ratio):
    df = pd.read_csv(path)
    df = df.dropna()
    df['RightGazeDirection'] = df['RightGazeDirection'].apply(lambda x: ast.literal_eval(x))
    df['Unit_Vector'] = df['Unit_Vector'].apply(lambda x: ast.literal_eval(x))
    gaze = df["RightGazeDirection"][::downsample_ratio].values.tolist()
    head = df["Unit_Vector"][::downsample_ratio].values.tolist()
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
                temp1, temp2 = preprocess_one_csv("../Data/Version2/" + per + "/" + csvs[f], dataset_cfg.seq_len, dataset_cfg.downsample_ratio)
                if temp1 is not None and temp2 is not None:
                    gaze = np.concatenate((gaze, temp1), axis=0)
                    head = np.concatenate((head, temp2), axis=0)
        except:
            pass
    assert head.shape == gaze.shape
    np.save("../LIMU-BERT-Public/dataset/hgbd/data_2.npy", gaze)
    np.save("../LIMU-BERT-Public/dataset/hgbd/label_2.npy", head)
    # np.save("../LIMU-BERT-Public/dataset/hgbd/label_2.npy", np.ones(gaze.shape))


def main(args, training_rate, tracker):
    # preprocess_hgbd_dataset(args)
    gdata, hdata, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    #Setting mask_cfg from hyperparameters:
    mask_cfg = MaskConfig(mask_ratio=args.D_MASKRATIO, 
                      mask_alpha=mask_cfg.mask_alpha, 
                      max_gram=mask_cfg.max_gram, 
                      mask_prob=mask_cfg.mask_prob, 
                      replace_prob=mask_cfg.replace_prob)

    #pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    pipeline = [Preprocess4Mask(mask_cfg)]
    gdata_train, hdata_train, gdata_val, hdata_val, gdata_test, hdata_test = prepare_pretrain_dataset(gdata, hdata, training_rate, seed=train_cfg.seed)

    model = None
    if args.model_type == 'gaze':
        dataset_pretrain = LIBERTGazeDataset4Pretrain
        model = LIMUBertAEModel4Pretrain(model_cfg)
        data_set_train = dataset_pretrain(gdata_train, pipeline=pipeline)
        data_set_val = dataset_pretrain(gdata_val, pipeline=pipeline)
        data_set_test = dataset_pretrain(gdata_test, pipeline=pipeline)
    elif args.model_type == 'gaze_mm':
        dataset_pretrain = LIBERTMultiDataset4Pretrain
        model = LIMUBertMultiMAEModel4Pretrain(model_cfg,recon_head=False)
    elif args.model_type == 'head_gaze_mm':
        dataset_pretrain = LIBERTMultiDataset4Pretrain
        model = LIMUBertMultiMAEModel4Pretrain(model_cfg,recon_head=True)

    if args.model_type != 'gaze':
        data_set_train = dataset_pretrain(gdata_train, hdata_train, pipeline=pipeline)
        data_set_val = dataset_pretrain(gdata_val, hdata_val, pipeline=pipeline)
        data_set_test = dataset_pretrain(gdata_test, hdata_test, pipeline=pipeline)
        
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_val = DataLoader(data_set_val, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)


    def func_loss(model, batch):
        if args.model_type == 'gaze':
            gmask_seqs, gmasked_pos, gseqs = batch
        else:    
            gmask_seqs, gmasked_pos, gseqs, hmask_seqs, hmasked_pos, hseqs = batch
        
        if args.model_type == 'gaze_mm':
            gseq_recon = model(gmask_seqs, hmask_seqs, gmasked_pos)
            gloss_lm = criterion(gseq_recon, gseqs)
            loss_lm = gloss_lm
        elif args.model_type == 'head_gaze_mm':
            gseq_recon, hseq_recon = model(gmask_seqs, hmask_seqs, gmasked_pos)
            gloss_lm = criterion(gseq_recon, gseqs)
            hloss_lm = criterion(hseq_recon, hseqs)
            #loss_lm = gloss_lm + hloss_lm
            loss_lm = torch.concat((gloss_lm,hloss_lm), dim = 1)
            #import pdb; pdb.set_trace()
        else:
            gmask_seqs, gmasked_pos, gseqs = batch
            gseq_recon = model(gmask_seqs, gmasked_pos)
            gloss_lm = criterion(gseq_recon, gseqs) 
            loss_lm = gloss_lm
        return loss_lm

    def func_forward(model, batch):
        if args.model_type == 'gaze_mm':
            gmask_seqs, gmasked_pos, gseqs, hmask_seqs, hmasked_pos, hseqs = batch
            gseq_recon = model(gmask_seqs, hmask_seqs, gmasked_pos)
            return gseq_recon, gseqs
        elif args.model_type == 'head_gaze_mm':
            gmask_seqs, gmasked_pos, gseqs, hmask_seqs, hmasked_pos, hseqs = batch
            gseq_recon, hseq_recon = model(gmask_seqs, hmask_seqs, gmasked_pos)
            #import pdb; pdb.set_trace();
            return torch.concat((gseq_recon,hseq_recon), dim = 1), torch.concat((gseqs,hseqs), dim = 1)
        else:
            gmask_seqs, gmasked_pos, gseqs = batch
            gseq_recon = model(gmask_seqs, gmasked_pos)
            return gseq_recon, gseqs
        
    def func_evaluate(seqs, gpredict_seqs):
        if args.model_type == 'gaze_mm':
            gloss_lm = criterion(gpredict_seqs, seqs)
            return gloss_lm.mean().cpu().numpy()
        elif args.model_type == 'head_gaze_mm':
            gloss_lm = criterion(gpredict_seqs, seqs)
            return gloss_lm.mean().cpu().numpy()
        else:
            gloss_lm = criterion(gpredict_seqs, seqs)
            return gloss_lm.mean().cpu().numpy()

    tracker.log_parameters(train_cfg, model_cfg, mask_cfg, dataset_cfg)
    
    if hasattr(args, 'pretrain_model'):
        val_loss, test_loss, train_loss = trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_val, data_loader_test
                    , model_file=args.pretrain_model)
    else:
        val_loss, test_loss, train_loss = trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_val, data_loader_test, model_file=None)

    tracker.log_model(model, "models")
    tracker.log_metrics("Train Loss", train_loss)
    tracker.log_metrics("Val Loss", val_loss)
    tracker.log_metrics("Test Loss", test_loss)
        
    gaze_estimate_test, gaze_actual_test = trainer.run(func_forward, None, data_loader_test, return_labels=True)

    return gaze_actual_test, gaze_estimate_test

if __name__ == "__main__":
    mode = "base"
    training_rate = 0.8

    tracker = tracking.MLFlowTracker("Head and Gaze Prediction ||")
    tracker.set_experiment()

    with mlflow.start_run(description="A MultiModal Transformer"):
        gaze_actual_test, gaze_estimate_test = main(args, training_rate, tracker)
        # tracker.log_metrics("Test Levenschtein Distance", compute_levenschtein_distance(gaze_estimate_test, gaze_actual_test))
        tracker.log_metrics("Test Dynamic Time Warping", compute_dtw_metric(gaze_estimate_test, gaze_actual_test))

        datestr = datetime.now().strftime("%d.%m.%Y.%H.%M")
        plot.plot3DLine(gaze_estimate_test, gaze_actual_test, "3DLine_", datestr)
        tracker.log_artifact(os.path.join(os.getcwd(), "results", f"3DLine_{datestr}.png"))
        tracker.log_artifact(args.save_path+'.pt')

