#!/usr/bin/env python3

try:
    import intro
except ModuleNotFoundError:
    import os,sys
    sys.path.insert(0,os.path.join(os.environ['VIRTUAL_ENV'],'src','intro_dev'))
    import intro

import pandas as pd
import numpy as np
import seaborn as sb
from umap import UMAP

import torchvision as tv
import matplotlib.pyplot as plt

from torchvision.datasets import KMNIST as MNIST
from torchvision import transforms as xfm

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("nn",default=15,type=int,help="Direct control of Nearest Neighbor size")
    p.add_argument("--seed",default=None,type=int,help="Set the seed")
    return p.parse_args()

def view_dataset(data:np.ndarray):
    print("Viewing the first 400 images in a dataset of shape:",data.shape)
    fig, ax_array = plt.subplots(20, 20,figsize=(12,12))
    axes = ax_array.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(data[i], cmap='gray')
    plt.setp(axes, xticks=[], yticks=[], frame_on=False)
    plt.tight_layout(h_pad=0.001, w_pad=0.001)
    # plt.show()

def make_df(dataset):
    data = dataset.data
    targets = dataset.targets.data
    print("Dataset dimensions:",data.shape)
    print("Target dimensions:",targets.shape)
    df = pd.DataFrame({"labels":targets.tolist(),'data':data.tolist(),"embed0":[0]*len(targets),"embed1":[0]*len(targets)})
    return df


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dataset = MNIST(intro.get_dataset_dir(),train=True)
    df = make_df(dataset)
    df = df.sample(frac=1.0,random_state=rng.integers(np.iinfo(np.int32).max,size=(10,)))
    view_dataset(np.stack(df['data'].values))


    digits = np.stack(df['data'].values).reshape((len(df),-1))
    labels = [dataset.classes[x] for x in df['labels'].values]
    df['labels'] = labels
    print("Flattened dataset shape:",digits.shape)
    # https://umap-learn.readthedocs.io/en/latest/basic_usage.html#digits-data
    reducer = UMAP(n_neighbors=args.nn,n_components=2,metric='euclidean',min_dist=0.1,verbose=True,low_memory=True,random_state=rng.integers(np.iinfo(np.int32).max))
    embeded_data = reducer.fit_transform(digits)
    print("Embedded dataset shape:",embeded_data.shape)

    df['embed0'] = embeded_data[:,0].tolist()
    df['embed1'] = embeded_data[:,1].tolist()
    print(df)

    fig,ax = plt.subplots(1,figsize=(16,12))
    sb.scatterplot(df,x="embed0",y="embed1",hue="labels",hue_order=dataset.classes,ax=ax)
    plt.show()






















if __name__ == '__main__':
    main()
