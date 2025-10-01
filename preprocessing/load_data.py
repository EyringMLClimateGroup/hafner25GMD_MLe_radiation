import xarray as xr
import numpy as np
from datetime import datetime
from pymeeus.Epoch import Epoch
from pymeeus.Earth import Earth
import xbatcher
from torch.utils.data import Dataset as TorchDataset, IterableDataset
import torch
import time


class PreBatchedDataset(IterableDataset):
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.bs = batch_size
        n_full = len(dataset) // batch_size
        self.batches = [
            dataset[i*batch_size:(i+1)*batch_size] for i in range(n_full)
        ] + [dataset[n_full*batch_size:]]


    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

class Np_Dataset(TorchDataset):
    def __init__(self, ds_path, variables, vweights=np.ones((47,81920)), testset=False):
        t0 = time.time()
        self.ds = xr.open_mfdataset(ds_path, engine="zarr")
        self.x_vars = variables["in_vars"]
        self.y_vars = variables["out_vars"] 
        self.extra_vars = variables["extra_in"]
        self.vweights = vweights
        if testset:
            self.x_data, self.y_data = self.prepare_for_testing(self.ds.isel({"time":0}))
        else:    
            self.x_data, self.y_data = self.prepare_dataset(self.ds)
        t1 = time.time()
        print("Dataset loading: ", t1-t0)
        
    def prepare_for_testing(self, ds):
        x_all = []
        for x in self.x_vars:
            v = ds[x].values
            if len(v.shape) == 2:
                v = v.T
            else:
                v = v[:,np.newaxis]
            x_all.append(v)
        x = np.hstack(x_all)
        
        y_all = []
        for y in self.y_vars:
            v = ds[y].values
            if len(v.shape) == 2:
                v = v.T
            else:
                v = v[:,np.newaxis]
            y_all.append(v)
        y = np.hstack(y_all)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def prepare_dataset(self, ds):
        x_all = []
        for x in self.x_vars:
            v = ds[x].values
            if len(v.shape) == 2:
                v = np.hstack(v)[:,np.newaxis]
            else:
                v = np.hstack(v).T
            x_all.append(v)
        x = np.hstack(x_all)
        y_all = []
        for y in self.y_vars:
            v = ds[y].values
            if len(v.shape) == 2:
                v = np.hstack(v)[:,np.newaxis]
            else:
                v = np.hstack(v).T
            y_all.append(v)
        y = np.hstack(y_all)
        
        e = []
        for ei in self.extra_vars:
            if ei == "q":
                v = np.hstack(self.get_q(ds)).T
            elif ei == "vweights":
                if "cell_idx" in ds:
                    cells = np.hstack(ds.cell_idx.astype("int32").values)
                else:
                    cells = np.concatenate([ds.cells.astype("int32").values]*len(ds.time))
                v = self.vweights[:,cells].T
            elif ei in ["rsdscs","rsutcs","rldscs","rlutcs"]:
                v = self.prepare_var(ei, ds).values
                v = np.hstack(v)[:,np.newaxis]                
            else:
                v = ds[ei].values
                if len(v.shape) == 2:
                    v = np.hstack(v)[:,np.newaxis]
                else:
                    v = np.hstack(v).T
            e.append(v)
        
        if len(e) > 0 :
            e = np.hstack(e)
            self.extra_shape = e.shape[-1]
            x = np.hstack([x,e])
        else:
            self.extra_shape = 0
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def prepare_var(self, v, ds):
        if v == "rsdscs":
            return ds["rsdcs"][:,-1,:]*ds["cosmu0_rt"]/ds["cosmu0"]
        elif v == "rsutcs": 
            return ds["rsucs"][:,0,:]*ds["cosmu0_rt"]/ds["cosmu0"]
        elif v == "rldscs":
            return ds["rldcs"][:,-1,:]
        elif v == "rlutcs": 
            return ds["rlucs"][:,0,:]
        else:
            raise ValueError(f"Don't know variable {v}")
    def __len__(self):
        return len(self.x_data)

    def get_q(self, batch):
        q = batch["extra_3d_clw"] + batch["extra_3d_cli"] + np.abs( batch["extra_3d_hus"] - np.mean(batch["extra_3d_hus"], axis=1))
        qs = np.sum(q, axis=1)
        normed_q = q/qs
        return normed_q.squeeze()
        
    
    def __getitem__(self, idx):
        #return torch.tensor(self.x_data[idx], dtype=torch.float32), torch.tensor(self.y_data[idx], dtype=torch.float32)
        return self.x_data[idx], self.y_data[idx]


