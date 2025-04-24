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

from torch import nn
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision import transforms as xfm

from intro.models import Classifier_v0,AutoEncoder_v0

# def view_dataset(data:np.ndarray):
#     print("Viewing the first 400 images in a dataset of shape:",data.shape)
#     fig, ax_array = plt.subplots(20, 20,figsize=(12,12))
#     axes = ax_array.flatten()
#     for i, ax in enumerate(axes):
#         ax.imshow(data[i], cmap='gray')
#     plt.setp(axes, xticks=[], yticks=[], frame_on=False)
#     plt.tight_layout(h_pad=0.001, w_pad=0.001)
#     # plt.show()

def make_df(dataset):
    data = dataset.data
    targets = dataset.targets.data
    print("Dataset dimensions:",data.shape)
    print("Target dimensions:",targets.shape)
    df = pd.DataFrame({"labels":targets.tolist(),'data':data.tolist(),"embed0":[0]*len(targets),"embed1":[0]*len(targets)})
    return df


def main():
    import lightning as lp
    rng = np.random.default_rng()
    dataset = MNIST(intro.get_dataset_dir(),train=True)
    df = make_df(dataset)
    df = df.sample(frac=1.0,random_state=rng.integers(np.iinfo(np.int32).max,size=(10,)))
    # view_dataset(np.stack(df['data'].values))
    cls_label = [x for x in dataset.classes]


    digits = np.stack(df['data'].values).reshape((len(df),-1))
    labels = [cls_label[x] for x in df['labels'].values]
    df['labels'] = labels
    print("Flattened dataset shape:",digits.shape)
    fig1,ax1 = plt.subplots(1,figsize=(16,12))
    reducer_seed = rng.integers(np.iinfo(np.int32).max)
    del rng
    reducer = UMAP(n_neighbors=15,n_components=2,metric='euclidean',min_dist=0.1,verbose=True,low_memory=True,random_state=reducer_seed)
    embeded_data = reducer.fit_transform(digits)
    del reducer
    print("Embedded dataset shape:",embeded_data.shape)

    df['embed0'] = embeded_data[:,0].tolist()
    df['embed1'] = embeded_data[:,1].tolist()
    sb.scatterplot(df,x="embed0",y="embed1",hue="labels",hue_order=[cls_label[x] for x in range(10)],ax=ax1)
    del df, embeded_data,digits,dataset


    # How do feature extraction routines compare
    ae_model = AutoEncoder_v0.load_from_checkpoint('lightning_logs/ae_mnist/stage_3/model.ckpt')
    cls_model = Classifier_v0.load_from_checkpoint('lightning_logs/cls_mnist/stage_3/model.ckpt')

    # Select the 'feature' extraction subnet
    ae_enc = ae_model.encoder
    cls_bb = cls_model.backbone


    class ae_flatter(nn.Module):
        def __init__(self,encoder):
            super().__init__()
            self.encoder = encoder
        def forward(self,x):
            return self.encoder(x.view(x.shape[0],-1))

    class infer_intf(lp.LightningModule):
        def __init__(self,models_contrast):
            super().__init__()
            self.models = nn.ModuleList(models_contrast)

        def predict_step(self, batch, batch_idx):
            observation,label = batch
            contrast = [m(observation) for m in self.models]
            return contrast,label

    class ds(object):
        def __init__(self,data,lbl):
            from argparse import Namespace
            self.data=data
            self.targets = Namespace(data=lbl)

    from torch.utils.data import DataLoader
    batchsize=200
    manager = lp.Trainer(accelerator='auto',enable_checkpointing=False)
    dataset = MNIST(intro.get_dataset_dir(),train=True,transform=xfm.ToTensor())
    nobs = len(dataset)
    dataloader = DataLoader(dataset,num_workers=7,batch_size=batchsize)
    predictions = manager.predict(infer_intf([ae_flatter(ae_enc),cls_bb]),dataloader)
    inference, labels = zip(*predictions)
    del predictions, manager, dataset, dataloader, DataLoader, lp
    enc_infer, bb_infer = zip(*inference)
    del inference
    ae_features = np.concatenate([x.numpy() for x in enc_infer]).reshape(nobs,-1)
    cf_features = np.concatenate([x.numpy() for x in bb_infer]).reshape(nobs,-1)
    predicted_lbl = np.concatenate([x.numpy() for x in labels]).reshape(nobs,-1)

    del enc_infer, bb_infer

    # print(ae_features.shape)
    # print(cf_features.shape)
    # print(predicted_lbl.shape)


    df_ae = make_df(ds(ae_features,predicted_lbl))
    df_cl = make_df(ds(cf_features,predicted_lbl))
    del predicted_lbl, ae_features, cf_features

    labels = [cls_label[x[0]] for x in df_ae['labels'].values.reshape(nobs,)]
    df_ae['labels'] = labels
    df_cl['labels'] = labels
    del labels
    # print(df_ae)
    # df_ae.to_hdf('ae_df.pd',key='results')
    # del df_ae
    # df_cl.to_hdf('cl_df.pd',key='results')
    # print(df_cl)
    # del df_cl
    # # https://umap-learn.readthedocs.io/en/latest/basic_usage.html#digits-datasrc/intro_dev/scripts/umap_mnist.py
    # reducers = [UMAP(n_neighbors=15,n_components=2,metric='euclidean',min_dist=0.1,verbose=True,low_memory=True,random_state=rng.integers(np.iinfo(np.int32).max)) for _ in range(2)]
    # digits = np.stack(df['data'].values).reshape((len(df),-1))
    reducer = UMAP(n_neighbors=15,n_components=2,metric='euclidean',min_dist=0.1,verbose=True,low_memory=True,random_state=reducer_seed)
    embeded_ae = reducer.fit_transform(np.stack(df_ae['data'].values).reshape((nobs,-1)))
    del reducer
    df_ae['embed0'] = embeded_ae[:,0].tolist()
    df_ae['embed1'] = embeded_ae[:,1].tolist()
    # print(df_ae)

    fig,ax = plt.subplots(2,1,figsize=(16,12))
    sb.scatterplot(df_ae,x="embed0",y="embed1",hue="labels",hue_order=[cls_label[x] for x in range(10)],ax=ax[0])
    del df_ae,embeded_ae

    reducer = UMAP(n_neighbors=15,n_components=2,metric='euclidean',min_dist=0.1,verbose=True,low_memory=True,random_state=reducer_seed)
    embeded_cl = reducer.fit_transform(np.stack(df_cl['data'].values).reshape((nobs,-1)))
    del reducer
    df_cl['embed0'] = embeded_cl[:,0].tolist()
    df_cl['embed1'] = embeded_cl[:,1].tolist()
    # print(df_cl)
    
    sb.scatterplot(df_cl,x="embed0",y="embed1",hue="labels",hue_order=[cls_label[x] for x in range(10)],ax=ax[1])
    del df_cl,embeded_cl
    plt.show()






















if __name__ == '__main__':
    main()
