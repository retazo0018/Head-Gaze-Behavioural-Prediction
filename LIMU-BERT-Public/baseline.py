from utils import handle_argv, load_pretrain_data_config, prepare_pretrain_dataset, Preprocess4Mask, LIBERTGazeDataset4Pretrain
from torch.utils.data import DataLoader
from scipy import interpolate
import numpy as np
import torch
import models, train, tracking, plot
import os 
from datetime import datetime
from statistic import compute_dtw_metric, compute_euclidean_distance
import math

def conv_spherical(seq):
    if not isinstance(seq, torch.Tensor):
        seq = torch.Tensor(seq)
    
    x = seq[..., 0]
    y = seq[..., 1]
    z = seq[..., 2]
    theta = torch.atan2(y, x)
    phi = torch.acos(z)
    theta[torch.isnan(theta)] = 0
    phi[torch.isnan(phi)] = 0
    spherical_coords = torch.stack((theta, phi), -1)
    return spherical_coords


mode = "base"
args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
training_rate = 0.8

gdata, hdata, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
gdata_train, hdata_train, gdata_val, hdata_val, gdata_test, hdata_test = prepare_pretrain_dataset(gdata, hdata, training_rate, seed=train_cfg.seed)

pipeline = [Preprocess4Mask(mask_cfg)]
# data_set_train = LIBERTGazeDataset4Pretrain(gdata_train, pipeline=pipeline)
data_set_test = LIBERTGazeDataset4Pretrain(gdata_test, pipeline=pipeline, istestset=True)

# data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=data_set_train.__len__())
data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=data_set_test.__len__())

seed = 2024

tracker = tracking.MLFlowTracker(f"Inference: Seed = {str(seed)}")
tracker.set_experiment()

# Apply interpolation to each sample in the array
gestimate = np.empty((0, dataset_cfg.seq_len, dataset_cfg.dimension))
for batch in data_loader_test:
    gseq_masked, gmasked_pos, gseqs = batch
    gseq_masked, gmasked_pos, gseqs = gseq_masked.cpu().detach().numpy(), gmasked_pos.cpu().detach().numpy(), gseqs.cpu().detach().numpy()
    for seq in range(gseq_masked.shape[0]):
        non_zero_indices = np.where((gseq_masked[seq] != 0).all(axis=1))[0]
        f = interpolate.interp1d(non_zero_indices, gseq_masked[seq][non_zero_indices], axis=0, kind='linear', fill_value="extrapolate")
        grecon = f(np.arange(len(gseq_masked[seq])))
        gestimate = np.concatenate((gestimate, np.expand_dims(grecon, axis=0)), axis=0)

estimate_test_sph, actual_test_sph = conv_spherical(gestimate), conv_spherical(gseqs[0:])
    
tracker.log_metrics("Test Euclidean Distance", compute_euclidean_distance(estimate_test_sph, actual_test_sph))
tracker.log_metrics("Test Dynamic Time Warping", compute_dtw_metric(estimate_test_sph, actual_test_sph))

datestr = datetime.now().strftime("%d.%m.%Y.%H.%M")
plot.plot_sequences_3d(estimate_test_sph, actual_test_sph, "3D_Spherical_Coord_Gaze_", datestr, seed=seed)
tracker.log_artifact(os.path.join(os.getcwd(), "results", f"3D_Spherical_Coord_Gaze_{datestr}.png"))
plot.plot_sequences_2d(estimate_test_sph, actual_test_sph, "2D_Spherical_Coord_Head_", datestr, seed=seed)
tracker.log_artifact(os.path.join(os.getcwd(), "results", f"2D_Spherical_Coord_Head_{datestr}.png"))
