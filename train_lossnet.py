from ast import arg
import source_2_loss as source
from source_2_loss import config
import os
import random
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split

#import segmentation_models_pytorch as smp
import tifffile
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

import glob
import argparse
parser = argparse.ArgumentParser(description='Loss Net')
parser.add_argument('--dataroot', type=str, default='./datasets/*/labels/*.tif')

# -----------------------
# --- Main parameters ---
# -----------------------
root = "data"
seed = 1
learning_rate = 1e-5
batch_size = 6
n_epochs = 150
classes = [1, 2, 3, 4, 5,6, 7, 8]
n_classes = len(classes) + 1
ngpu = torch.cuda.device_count()

outdir = "weights"
os.makedirs(outdir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -------------------------------------------
# --- extract datasets ---
# -------------------------------------------
fns_list = glob.glob(arg.dataroot)

# ----------------------------------
# Calcurate the weight of each class
# ----------------------------------
num_pixcel = np.zeros(n_classes)
for idx in tqdm(range(len(fns_list))):
    uni, count = np.unique(tifffile.imread(fns_list[idx]), return_counts=True)
    for i, u in enumerate(uni):
        num_pixcel[u] += count[i]
classes_wt = np.asarray(np.reciprocal(num_pixcel)*1e7, dtype=np.float32)
print("classes_weight: ", classes_wt)

# ----------------------------------
# --- Define dataset and loaders ---
# ----------------------------------
trainset = source.dataset.Dataset(fns_list, classes=classes, train=True)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

# --------------------------
#       network setup
# --------------------------
network = source.unet.UNet(in_channels=3, classes=n_classes)
lossnet = source.lossnet.LossNet()
if (device.type == 'cuda') and (ngpu > 1):
    network = nn.DataParallel(network, list(range(ngpu)))
    lossnet = nn.DataParallel(lossnet, list(range(ngpu)))
    
models = {'backbone': network, 'module': lossnet}
criterion = source.losses.CEWithLogitsLoss(weights=classes_wt)

metric = source.metrics.IoU()
optim_backbone = torch.optim.Adam(models['backbone'].parameters(), lr=learning_rate)
optim_module = torch.optim.Adam(models['module'].parameters(), lr=learning_rate)

optimizers = {'backbone': optim_backbone, 'module': optim_module}
network_fout = f"unet_s{seed}_{criterion.name}"
#lossnet_fout = f"{lossnet.name}_s{seed}_{criterion.name}"
#print("Model output name  :", network_fout, ", ", lossnet_fout)



# ------------------------
# --- Network training ---
# ------------------------
start = time.time()
train_hist = []
valid_hist = []
train_total_loss = []
train_module_loss = []
predicted_loss = []

for epoch in range(n_epochs):
    print(f"\nEpoch: {epoch + 1}")
    # if the number of data is odd number, the model does not work  
    if len(fns_list) %2 == 1:
        even_file = random.sample(fns_list, len(fns_list)-1)
        trainset = source.dataset.Dataset(even_file, classes=classes, train=True)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    logs_train, total_loss, module_loss = source.runner.train_epoch(
        models=models,
        optimizers=optimizers,
        criterion=criterion,
        metric=metric,
        dataloader=train_loader,
        device=device,
        epoch = epoch,
    )
    train_total_loss.append(total_loss)
    train_module_loss.append(module_loss)
    
    train_hist.append(logs_train)

    torch.save({'epoch': epoch + 1, 'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()}, \
                   os.path.join(outdir, f"{network_fout}.pth"))
    print(total_loss, "Model saved!")

    # do something
    
    if epoch % 30 == 0 and epoch != 0:
        learning_rate *= 5e-1
        optimizers['backbone'].param_groups[0]["lr"] = learning_rate
        optimizers['module'].param_groups[0]["lr"] = learning_rate
        print("Decrease decoder learning rate to {}".format(learning_rate))


print(f"Completed: {(time.time() - start)/60.0:.4f} min.")