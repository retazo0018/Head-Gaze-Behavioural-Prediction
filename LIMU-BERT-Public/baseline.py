from utils import handle_argv, load_pretrain_data_config, prepare_pretrain_dataset, Preprocess4Mask, LIBERTGazeDataset4Pretrain
from torch.utils.data import DataLoader
from scipy import interpolate
import numpy as np


mode = "base"
args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
training_rate = 0.8

gdata, hdata, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
gdata_train, hdata_train, gdata_val, hdata_val, gdata_test, hdata_test = prepare_pretrain_dataset(gdata, hdata, training_rate, seed=train_cfg.seed)

pipeline = [Preprocess4Mask(mask_cfg)]
data_set_train = LIBERTGazeDataset4Pretrain(gdata_train, pipeline=pipeline)
data_set_test = LIBERTGazeDataset4Pretrain(gdata_test, pipeline=pipeline)

data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

# Apply interpolation to each sample in the array
gestimate = np.empty((0, dataset_cfg.seq_len, dataset_cfg.dimension))
for batch in data_loader_test:
    gseq_masked, gmasked_pos, gseqs = batch
    gseq_masked, gmasked_pos, gseqs = gseq_masked.cpu().detach().numpy(), gmasked_pos.cpu().detach().numpy(), gseqs.cpu().detach().numpy()
    for seq in range(gseq_masked.shape[0]):
        non_zero_indices = np.where((gseq_masked[seq] != 0).all(axis=1))[0]
        f = interpolate.interp1d(non_zero_indices, gseq_masked[seq][non_zero_indices], axis=0, kind='cubic', fill_value="extrapolate")
        grecon = f(np.arange(len(gseq_masked[seq])))
        gestimate = np.concatenate((gestimate, np.expand_dims(grecon, axis=0)), axis=0)