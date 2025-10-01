import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
import xarray as xr
import numpy as np
from glob import glob
import random
from tqdm import tqdm
from numba import njit
import xbatcher
from torch.utils.data import Dataset as TorchDataset
import torch
import yaml
import config
from joblib import Parallel, delayed
import argparse

import matplotlib.pyplot as plt
random.seed(3141)
np.random.seed(3141)

zh_lr_path = "/p/project1/icon-a-ml/hafner1/org_radiation/preprocessing/atm_amip_R2B5_vgrid_ml.nc"
zh_hr_path = "/p/project1/icon-a-ml/hafner1/zhalf_192.nc"
zf_hr_path = "/p/home/jusers/hafner1/juwels/project/hafner1/zfull_R02B05_191.nc"
#coarse = "/p/largedata2/icon-a-ml-data/qubicc_processed_data/coarse_grained/"
month = "july"
coarse = f"/p/scratch/icon-a-ml/savre-piou1/postprocessed/atm_qubicc_R02B09_{month}/"
high_res = "/p/largedata2/icon-a-ml-data/qubicc_processed_data/high_res/"
#preprocessed_data_path = "/p/project1/icon-a-ml/hafner1/cloudy_radiation/data/"
preprocessed_data_path = f"/p/project1/icon-a-ml/hafner1/cloudy_radiation/data/{month}/"
zh_hr_ds = xr.open_mfdataset(zh_hr_path)
zh_hr = zh_hr_ds["zhalf"].values
zh_lr_ds = xr.open_mfdataset(zh_lr_path)
zh_lr = zh_lr_ds["zhalf"].values
zf_lr = zh_lr_ds["zfull"].values
zf_hr_ds = xr.open_dataset(zf_hr_path)
zf_hr = zf_hr_ds["zfull"].values
print("Z ", zh_lr[0], zf_lr[0], zh_hr[0], zf_hr[0])


def vert_coarse_grain_fl(var, zh_lr_vals, zh_hr_vals):
    """
    vertically coarse graining a variable based on https://github.com/EyringMLClimateGroup/heuer23_ml_convection_parameterization/blob/main/preprocessing/VerticalCoarseGrainingParallel.py
    coarse grained using weighted average
    
    Args:
        var: variable to vertically coarse grain. Shape: (time, height, cell)
        zh_lr_vals: height at half levels of low resolution (height, cell)
        zh_hr_vals: height at half levels of high resolution (height, cell)
    
    Return:
        result: vertically coarse grained var
    """
    # create array of target shape
    result = np.full((var.shape[0], zh_lr_vals.shape[0]-1, var.shape[2]), np.nan) 
    # iterate through low resolution height levels
    for j in range(zh_lr_vals.shape[0]-1): 
        # note: ICON starts at the top --> lower bound at j+1
        z_u = zh_lr_vals[j,:]     # upper bound of current height level
        z_l = zh_lr_vals[j+1,:]   # lower bound of current height level 
        # get z_l <= zh_hr <= z_u  
        # 1. get all hr values below upper bound
        all_upper_below_u = np.minimum(z_u, zh_hr_vals[:-1])
        # 2. get all hr vlues above lower bound
        all_lower_above_l = np.maximum(zh_hr_vals[1:], z_l)
        # 3. take the difference
        diff = all_upper_below_u - all_lower_above_l
        # diff is negative for all levels above u (z_u - zh_hr_above_u) 
        # and also negative for all levels below l (zh_hr_below_l - z_l)       
        weights = np.maximum(diff, 0)
        # calculate weighted average
        result[0,j,:] = np.nansum(np.multiply(weights, var[0]), axis=0)/(z_u - z_l)

        # remove columns with height mismatch larger than 0.5m
        # when does this happen? 
        result[0,j] = np.where((np.abs((z_u - z_l) - np.sum(weights, axis = 0))) >= 0.5, np.nan, result[0,j])

    return result


def vert_coarse_grain_hl_from_half(var, zh_lr_vals, zf_hr_vals):
    result = np.full((var.shape[0], zh_lr_vals.shape[0], var.shape[2]), np.nan)
    result[:,0,:] = var[:,0] # Boundary condition for half lvl 
    result[:,-1,:] =  var[:,-1] # Boundary condition for half lvl 

    zf_lr_vals = (zh_lr_vals[1:,:] + zh_lr_vals[:-1,:]) / 2
    for j in range(zf_lr_vals.shape[0]-1):
        z_u = zf_lr_vals[j,:]
        z_l = zf_lr_vals[j+1,:]
        weights = np.maximum(np.minimum(z_u, zf_hr_vals[:-1]) - np.maximum(zf_hr_vals[1:], z_l), 0)

        # Following implementations are equivalent
        # result[0,j+1,:] = np.einsum('ij,ij->j', weights, var[0,1:-1])/(z_u - z_l)
        # result[0,j+1,:] = np.diag(weights.T @ var[0,1:-1])/(z_u - z_l)
        # result[0,j+1,:] = np.array([np.sum(np.array([weights[k,l]*var[0,k+1,l] for k in range(weights.shape[0])])) for l in range(var.shape[2])])/(z_u - z_l)
        result[0,j+1,:] = np.nansum(np.multiply(weights, var[0, 1:-1]), axis=0)/(z_u - z_l)

        result[0,j+1] = np.where((np.abs((z_u - z_l) - np.sum(weights, axis = 0))) >= 0.5, np.nan, result[0,j+1])

    return result


def vert_coarse_grain_hl_from_full(var, surf_var, zh_lr_vals, zf_hr_vals):
    result = np.full((var.shape[0], zh_lr_vals.shape[0], var.shape[2]), np.nan)
    result[:,0,:] = var[:,0] # Boundary condition for half lvl 
    result[:,-1,:] = surf_var # Boundary condition for half lvl 

    zf_lr_vals = (zh_lr_vals[1:,:] + zh_lr_vals[:-1,:]) / 2
    for j in range(zf_lr_vals.shape[0]-1):
        z_u = zf_lr_vals[j,:]
        z_l = zf_lr_vals[j+1,:]
        weights = np.maximum(np.minimum(z_u, zf_hr_vals[:-1]) - np.maximum(zf_hr_vals[1:], z_l), 0)

        # Following implementations are equivalent
        # result[0,j+1,:] = np.einsum('ij,ij->j', weights, var[0,1:-1])/(z_u - z_l)
        # result[0,j+1,:] = np.diag(weights.T @ var[0,1:-1])/(z_u - z_l)
        # result[0,j+1,:] = np.array([np.sum(np.array([weights[k,l]*var[0,k+1,l] for k in range(weights.shape[0])])) for l in range(var.shape[2])])/(z_u - z_l)
        result[0,j+1,:] = np.nansum(np.multiply(weights, var[0]), axis=0)/(z_u - z_l)

        result[0,j+1] = np.where((np.abs((z_u - z_l) - np.sum(weights, axis = 0))) >= 0.5, np.nan, result[0,j+1])

    return result    
class XBatcherPyTorchDataset(TorchDataset):
    def __init__(self, ds, variables, height=47, batch_size=1, cache=None):
        self.bgen = xbatcher.BatchGenerator(ds,
                              input_dims = {"height": height, "cell": 1}, 
                              batch_dims = {"time":1,"cell": batch_size},
                              concat_input_dims=True, preload_batch=False
                              )
        self.batch_size=batch_size
        self.x_vars = variables["in_vars"]
        self.y_vars = variables["out_vars"] 
        self.extra_vars = variables["extra_in"]
        self.cache = cache if cache is not None else dict()
        
    def __len__(self):
        return len(self.bgen)

    def get_q(self, batch):
        q = batch["extra_3d_clw"] + batch["extra_3d_cli"] + np.abs( batch["extra_3d_hus"] - np.mean(batch["extra_3d_hus"], axis=1))
        qs = np.sum(q, axis=1)
        normed_q = q/qs
        return normed_q.squeeze()

    def norm(self, batch, v):
        if v in ["tend_ta_rlw", "tend_ta_rsw"]:
            d = batch[v].squeeze().values
            d = d*86400
        elif v in ["rlu", "rld", "rlut", "rlus", "rlds", "rlds_rld"]:
            sig = 5.670374419e-8
            d = batch[v]
            ta = batch["ts_rad"]
            d = d/(ta**4*sig)
            d = d.values
        elif v in ["rsu", "rsd", "rsus", "rsut", "rsds", "rvds_dir", "rvds_dif", "rpds_dir", "rpds_dif", "rnds_dir", "rnds_dif"]:
            d = batch[v].squeeze().values
            rsdt = batch["toa"].squeeze().values # toa
            if v == "rsus":
                d = np.where(rsdt>0,d/rsdt, 0) 
            elif v == "rsut":
                d = np.where(rsdt>0, d/rsdt,0)
            elif v == "rsds":
                d = np.where(rsdt>0,d/rsdt, 0) 
            elif v  in [ "rvds_dir", "rvds_dif", "rpds_dir", "rpds_dif", "rnds_dir", "rnds_dif"]:
                d = np.where(batch["rsds"].squeeze()>0, d/rsdt, 0)
        else:
            d = batch[v].values
        return d.squeeze()
    
    def preprocess_batch(self, batch):
        
        x = []
        for xi in self.x_vars:
            if len(batch[xi].shape)==3:
                x.append(batch[xi].squeeze())
            elif batch[xi].shape[0]==1:
                x.append(batch[xi][0])
            else:
                x.append(batch[xi])
        x = np.hstack(x)
        
        y = [] 
        for yi in self.y_vars:
            v = self.norm(batch, yi)
            if len(v.shape)==1 and self.batch_size>1:
                v = v[:,np.newaxis]
            y.append(v)
        y = np.hstack(y)

        e = []
        for ei in self.extra_vars:
            if ei == "q":
                e.append(self.get_q(batch))
            else:
                v = self.norm(batch, ei)
                if len(v.shape)==1 and self.batch_size>1:
                    v = v[:,np.newaxis]
                e.append(v)
        
        if len(e) > 0 :
            e = np.hstack(e)
            self.extra_shape = e.shape[-1]
            x = np.hstack([x,e])
        else:
            self.extra_shape = 0
        return x, y

    def __getitem__(self, idx):
        if idx in self.cache.keys():
            return self.cache[idx]
        else:
            batch = self.bgen[idx].load()    
            x, y = self.preprocess_batch(batch)
            self.cache[idx] = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
def get_cvair(coarse, t):
    """
    from ICON: https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2025.04-public/src/atm_phy_aes/mo_aes_phy_diag.f90
        qliq = field%qtrc_phy(jc,jk,jb,iqc) + field%qtrc_phy(jc,jk,jb,iqr)
        qice = field%qtrc_phy(jc,jk,jb,iqi) + field%qtrc_phy(jc,jk,jb,iqs) + field%qtrc_phy(jc,jk,jb,iqg)
        qvap = field%qtrc_phy(jc,jk,jb,iqv)
        qtot = qvap + qice + qliq
        cv   = cvd*(1.0_wp - qtot) + cvv*qvap + clw*qliq + ci*qice
        field%cvair(jc,jk,jb) = cv*field%mair(jc,jk,jb)

    constants from: https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2025.04-public/src/shared/mo_physical_constants.f90
    """
    cvair_file = glob(f"{coarse}*cvair*{t}.nc")
    if len(cvair_file) > 0:
        print("cvair from file")
        cvair =  xr.open_mfdataset(cvair_file)["param255.0.0"]
        return cvair
    else:
        dt=40
        extra_qc = xr.open_mfdataset(glob(f"{coarse}*extra_3d_qc*{t}.nc"))["clwmr"]
        tend_qc_mig = xr.open_mfdataset(glob(f"{coarse}*tend_qc_mig*{t}.nc"))["param203.6.0"]
        qc = xr.align(extra_qc, tend_qc_mig, join="outer", fill_value=0)[0] + tend_qc_mig*dt
        extra_qr = xr.open_mfdataset(glob(f"{coarse}*extra_3d_qr*{t}.nc"))["rwmr"]
        tend_qr_mig = xr.open_mfdataset(glob(f"{coarse}*tend_qr_mig*{t}.nc"))["code255"]
        qr = xr.align(extra_qr, tend_qc_mig, join="outer", fill_value=0)[0] + tend_qr_mig*dt
        extra_qi = xr.open_mfdataset(glob(f"{coarse}*extra_3d_qi*{t}.nc"))["icmr"]
        tend_qi_mig = xr.open_mfdataset(glob(f"{coarse}*tend_qi_mig*{t}.nc"))["param213.6.0"]
        qi = xr.align(extra_qi, tend_qc_mig, join="outer", fill_value=0)[0] + tend_qi_mig*dt
        extra_qs = xr.open_mfdataset(glob(f"{coarse}*extra_3d_qs*{t}.nc"))["snmr"]
        tend_qs_mig = xr.open_mfdataset(glob(f"{coarse}*tend_qs_mig*{t}.nc"))["code255"]
        qs = xr.align(extra_qs, tend_qc_mig, join="outer", fill_value=0)[0] + tend_qs_mig*dt
        extra_qg = xr.open_mfdataset(glob(f"{coarse}*extra_3d_qg*{t}.nc"))["grle"]
        tend_qg_mig = xr.open_mfdataset(glob(f"{coarse}*tend_qg_mig*{t}.nc"))["code255"]
        qg = xr.align(extra_qg, tend_qc_mig, join="outer", fill_value=0)[0] + tend_qg_mig*dt
        extra_qv = xr.open_mfdataset(glob(f"{coarse}*extra_3d_qv*{t}.nc"))["q"]
        tend_qv_mig = xr.open_mfdataset(glob(f"{coarse}*tend_qv_mig*{t}.nc"))["param203.1.0"]
        qv = extra_qv + tend_qv_mig*dt
        mair = xr.open_mfdataset(glob(f"{coarse}*mair*{t}.nc"))["tcond"]
        qliq = qc + qr 
        qice = qi + qs + qg
        qvap = qv
        qtot = qvap + qice + qliq
        cpd  = 1004.64
        rv   = 461.51
        cvd  = cpd - 287.04 # cpd - rd 
        cvv  = 1869.46 - rv # cpv - rv
        clw  = (3.1733 + 1) * cpd # (rcpl + 1)*cpd
        ci   = 2106.0
        cv   = cvd*(1.0 - qtot) + cvv*qvap + clw*qliq + ci*qice 
        cvair = cv*mair
    return cvair

def loop_over_dates(d, variables_3d, variables_2d, sw=False, random_filter=True, partition="", model ="", args={}, tend_cs_nn=False):
        model = torch.jit.load(model)
        dataset_list = []
        files_2d = glob(f"{coarse}*atm2d*{d}.nc")
        data_2d = xr.open_mfdataset(files_2d)
        data_2d_sel = data_2d[variables_2d]
        data_2d.close()
        dataset_list.append(data_2d_sel)
        
        for k, v in variables_3d.items():
            print(k)
            file_3d = glob(f"{coarse}*{k}*{d}.nc")
            print(file_3d)
            data_3d = xr.open_mfdataset(file_3d)
            data_3d_sel = data_3d[v[0]]
            if v[0] in ["ccl", "icmr", "clwmr", "snmr"]: # variable too short. need broadcasting to match the shape for coarse graining
                broadcast_file = glob(f"{coarse}*o3*{d}.nc") # broadcast to any variable with full length
                broadcast_to_this = xr.open_mfdataset(broadcast_file)
                data_3d_sel, _ =  xr.align(data_3d_sel, broadcast_to_this, join="outer", fill_value=0)
                broadcast_to_this.close()
                
            if "extra" in k:
                # add microphysics tendencie sto extra 3d variables (after dynamics)
                tend_file_3d = glob(f"{coarse}*{v[-1]}*{d}.nc")
                tend_data_3d_ds = xr.open_mfdataset(tend_file_3d)
                tend_data_3d = tend_data_3d_ds[v[2]].values
                data_3d_sel = data_3d_sel+40*tend_data_3d            
                tend_data_3d_ds.close()
            if k == "rsd_":
                # flux profiles are saved at radiation time step and boundary fluxes at current time step
                # If we want to use the boundary fluxes from flux profiles then we need to scale them.
                # from ICON code: https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2025.04-public/src/atm_phy_aes/mo_radheating.f90?ref_type=heads
                # scaling factor xsdt is used to scale from flux profiles to boundary fluxes eg.
                # rsds = rsd_rt[-1] * xsdt
                # so we get xsdt by calculating
                # xsdt = rsds/rsd_rt[-1]
                # scaling for SW includes mainly change in solar zenith angle
                var = "toa"
                rsds = data_2d_sel["rsds"].values
                rsds_rt = data_3d_sel[:,-1].values # bottom entry of rsd is surface
                scaling = np.divide(rsds, rsds_rt, where=rsds_rt != 0, out= np.zeros_like(rsds_rt))
                sel_var = data_3d_sel[:,0]  # top entry of rsd is TOA
                sel_var = sel_var*scaling
                ds = xr.DataArray(
                        sel_var.values, 
                        name=var,
                        coords=[data_2d_sel.time.values, data_2d_sel.cell.values],
                        dims=["time", "cell"],
                    )
            elif k == "rld_":
                # for LW fluxes only rlus is scaled, so no scaling here 
                var = "rlds_rld" 
                sel_var = data_3d_sel[:,-1] # last entry of rld is surface
                ds = xr.DataArray(
                        sel_var.values, 
                        name=var,
                        coords=[data_2d_sel.time.values, data_2d_sel.cell.values],
                        dims=["time", "cell"],
                    )
            elif k == "phalf":
                vertically_coarse_grained = vert_coarse_grain_hl_from_full(data_3d_sel.values, data_2d_sel["ps"].values, zh_lr, zh_hr)
                ds = xr.DataArray(
                        vertically_coarse_grained,
                        name = v[1],
                        coords=[data_2d_sel.time, zh_lr_ds.height_2.values, data_2d_sel.cell],
                        dims=["time", "height_2", "cell"],
                        
                    )
            elif v[1] == "temp_level":
                vertically_coarse_grained = vert_coarse_grain_hl_from_full(data_3d_sel.values, data_2d_sel["ts"].values, zh_lr, zh_hr)
                ds = xr.DataArray(
                        vertically_coarse_grained,
                        name = v[1],
                        coords=[data_2d_sel.time, zh_lr_ds.height_2.values, data_2d_sel.cell],
                        dims=["time", "height_2", "cell"],
                        
                    )
            elif k in ["_rld_", "_rldcs_", "_rsd_", "_rsdcs_", "_rlu_", "_rlucs_", "_rsu_", "_rsucs_"]:
                vertically_coarse_grained = vert_coarse_grain_hl_from_half(data_3d_sel.values, zh_lr, zf_hr)
                ds = xr.DataArray(
                        vertically_coarse_grained,
                        name = v[1],
                        coords=[data_2d_sel.time, zh_lr_ds.height_2.values, data_2d_sel.cell],
                        dims=["time", "height_2", "cell"],
                        
                    )
            else:
                vertically_coarse_grained = vert_coarse_grain_fl(data_3d_sel.values, zh_lr, zh_hr)
                ds = xr.DataArray(
                        vertically_coarse_grained,
                        name = v[1],
                        coords=[data_2d_sel.time, zh_lr_ds.height.values, data_2d_sel.cell],
                        dims=["time", "height", "cell"],
                        
                    ) 
            data_3d.close()
            dataset_list.append(ds)
        
        cvair = get_cvair(coarse, d)
        cvair_cg = vert_coarse_grain_fl(cvair.values, zh_lr, zh_hr)
        ds = xr.DataArray(
                cvair_cg,
                name = "cvair",
                coords=[data_2d_sel.time, zh_lr_ds.height.values, data_2d_sel.cell],
                dims=["time", "height", "cell"],
            )
        dataset_list.append(ds)
        
        
        if sw:
            mode = "SW"
        else:
            mode = "LW"
        if tend_cs_nn == False:
            if mode == "SW":
                # see above for comment on scaling
                rsds = data_2d_sel["rsds"].values
                file_3d = glob(f"{coarse}*rsd_*{d}.nc")
                print(file_3d)
                data_3d = xr.open_mfdataset(file_3d)
                data_3d_sel = data_3d["dswrf"]
                rsds_rt = data_3d_sel[:,-1].values # bottom entry of rsd is surface
                scaling = np.divide(rsds, rsds_rt, where=rsds_rt != 0, out= np.zeros_like(rsds_rt))
                rucs = xr.open_mfdataset(glob(f"{coarse}*rsucs*{d}.nc"))["uswrf"]*scaling
                rdcs = xr.open_mfdataset(glob(f"{coarse}*rsdcs*{d}.nc"))["dswrf"]*scaling
                name="tend_ta_rsw_cs"
            elif mode == "LW":
                # correct upward flux  ar surface for changed surface temperature 
                # see https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2025.04-public/src/atm_phy_aes/mo_radheating.f90?ref_type=heads
                # drlus_dtsr = emiss(jc)*4._wp*stbo*tsr(jc)**3 ! derivative
                # dtsr       = tsr(jc) - tsr_rt(jc)            ! change in tsr
                # rlus(jc)   = rlu_rt(jc,klevp1)             & ! rlus = rlus_rt
                #     &          +drlus_dtsr * dtsr           !       + correction
                e = 0.996
                sig = 5.670374419e-8
                drlus_dtsr = e*sig*4*data_2d_sel["ts"].values**3
                dtsr = data_2d_sel["ts"].values-data_2d_sel["ts_rad"].values
                rucs = xr.open_mfdataset(glob(f"{coarse}*rlucs*{d}.nc"))["ulwrf"]
                rdcs = xr.open_mfdataset(glob(f"{coarse}*rldcs*{d}.nc"))["dlwrf"]
                rlus = rucs[:,-1,:] + drlus_dtsr*dtsr
                rucs[:,-1,:] = rlus
                name="tend_ta_rlw_cs"
            rncs = (rdcs-rucs).values
            qrw = rncs[:,:-1]-rncs[:,1:]
            tend_ta_cs = qrw/cvair
            tend_ta_cs_cg = vert_coarse_grain_fl(tend_ta_cs.values, zh_lr, zh_hr)
            print(tend_ta_cs_cg)
            ds = xr.DataArray(
                        tend_ta_cs_cg,
                        name = name,
                        coords=[data_2d_sel.time, zh_lr_ds.height.values, data_2d_sel.cell],
                        dims=["time", "height", "cell"],
                    )
            dataset_list.append(ds)
        dataset_list.append(zh_lr_ds[["zhalf", "zfull", "dzghalf"]].drop_vars(["clat", "clon"]).rename({"ncells": "cell"}).expand_dims(dim="time"))
        
        merged = xr.merge(dataset_list).dropna("cell")    
        print("droped:", (1-len(merged.cell)/81290)*100, "%")
        if sw:
            merged = merged.where(merged["cosmu0"].compute()>0, drop=True)    
        # filter clear? at least some?   
        if partition == "test":
            n_samples = 35000
            #n_samples = np.min((n_smaples, len(merged.cell))
            if not sw:
                n_samples = 74000 # remove the badly coarse grained data
        elif partition == "val":
            n_samples = 5000
        else:
            n_samples = 14000
        #if random_filter:
        # select random cells: make cell_idx (=random_cells) a variable and let cell be np.arange(0, len(random_cells))
        random_cells = np.random.choice(merged.cell, size=n_samples, replace=False)
        cells_da = xr.DataArray(
            random_cells[np.newaxis,:], 
            name="cell_idx",
            coords=[merged.time, np.arange(0,len(random_cells))],
            dims=["time", "cell"],
        )  
        random_selection = merged.sel({"cell": random_cells})
        random_selection["cell"] = np.arange(0,len(random_cells))
        rd_sel_with_idx = xr.merge([random_selection, cells_da])
        #else:
        #    n_samples = len(merged.cell)
        #    rd_sel_with_idx = merged        

        if tend_cs_nn == True:
            data =  XBatcherPyTorchDataset(rd_sel_with_idx, args["variables"], batch_size=n_samples)
            x, y = data.__getitem__(0)
            x[:,0:47*2] = torch.zeros_like(x[:,0:47*2]) # set cli, clw to zero
            if mode == "SW":
                x[:,47*4+2:47*5+2] = torch.zeros_like(x[:,47*4+2:47*5+2])                   # set cl to zero
                name="tend_ta_rsw_cs"
            elif mode == "LW":
                x[:,47*4+1:47*5+1] = torch.zeros_like(x[:,47*4+1:47*5+1])                   # set cl to zero
                name="tend_ta_rlw_cs"
            else:
                print(f"unknown mode type {mode}. Choose SW or LW ")
           
            tend_ta_clear = model(x).numpy()/86400   # K/d -> K/s
            print(tend_ta_clear)
            ds = xr.DataArray(
                        tend_ta_clear[:,:47].T[np.newaxis,:,:],
                        name = name,
                        coords=[rd_sel_with_idx.time, rd_sel_with_idx.height.values, rd_sel_with_idx.cell.values],
                        dims=["time", "height", "cell"],
                    )
            rd_sel_with_idx = xr.merge([rd_sel_with_idx,ds])
        return rd_sel_with_idx.drop_vars(["clat", "clon"])

def create_zarr(dates, variables_3d, variables_2d, sw=False, random_filter=True, partition="", model ="", args={}):    
      
    selections = Parallel(n_jobs=-1)(
        delayed(loop_over_dates)(
            d, variables_3d, variables_2d, sw, random_filter, partition, model, args
            ) for d in tqdm(dates))
    all_selections = xr.merge(selections)
    
    mode="SW" if sw else "LW"
    all_selections.to_zarr(f"{preprocessed_data_path}{partition}_{mode}.zarr", mode="w")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default='SW', choices=['SW', 'LW'])
    parser.add_argument("-p", "--partition", default='train', choices=['train', 'val', "test", "all"])
    parser = parser.parse_args()
    print(parser.mode, parser.partition)
    filenames = sorted(glob(f"{coarse}*rldcs*.nc"))
    dates = [filename.split('_')[-1].split('.')[0] for filename in filenames]
    n = len(dates)
    step = 1 # 4
    train_dates = dates[1:int(0.6*n):step]
    val_dates = dates[int(0.6*n+1):int(0.8*n):step]
    test_dates = dates[int(0.8*n+1)::step]
    print(len(train_dates), len(val_dates), len(test_dates))
    variables_3d_sw = {"o3": ["o3", "o3"], 
                "3d_den": ["den", "rho"], 
                "_cl_": ["ccl", "cl"], 
                "tend_ta_rsw":["param205.0.0", "tend_ta_rsw"],
                "extra_3d_qi_ml": ["icmr", "extra_3d_cli", "param213.6.0", "tend_qi_mig" ],
                "extra_3d_qc_ml": ["clwmr", "extra_3d_clw", "param203.6.0", "tend_qc_mig"],
                "rsd_": ["dswrf", "toa"],
                "extra_3d_ta_": ["t", "extra_3d_ta", "param203.0.0", "tend_ta_mig"],
                "extra_3d_qv_ml": ["q", "extra_3d_hus", "param203.1.0", "tend_qv_mig" ],
                "extra_3d_qs_ml": ["snmr", "extra_3d_qs", "code255", "tend_qs_mig" ],
                "pfull": ["pres", "pres_layer"],
                "phalf": ["pres", "pres_level"],
                "_extra_3d_ta_": ["t", "temp_level", "param203.0.0", "tend_ta_mig"],
                "_rsd_": ["dswrf", "rsd"],
                "_rsdcs_": ["dswrf", "rsdcs"],  
                "_rsu_": ["uswrf", "rsu"],
                "_rsucs_": ["uswrf", "rsucs"],  
                      }
    variables_3d_lw = {"o3": ["o3", "o3"], 
                    "3d_den": ["den", "rho"], 
                    "_cl_": ["ccl", "cl"], 
                    "tend_ta_rlw": ["param204.0.0", "tend_ta_rlw"], 
                    "extra_3d_qi_ml": ["icmr", "extra_3d_cli", "param213.6.0", "tend_qi_mig" ],
                    "extra_3d_qc_ml": ["clwmr", "extra_3d_clw", "param203.6.0", "tend_qc_mig"],
                    "rld_": ["dlwrf", "rlds_rld"],
                    "extra_3d_ta_": ["t", "extra_3d_ta", "param203.0.0", "tend_ta_mig"],
                    "extra_3d_qv_ml": ["q", "extra_3d_hus", "param203.1.0", "tend_qv_mig" ],
                    "extra_3d_qs_ml": ["snmr", "extra_3d_qs", "code255", "tend_qs_mig" ],
                    "_rld_": ["dlwrf", "rld"],
                    "_rldcs_": ["dlwrf", "rldcs"],
                    "_rlu_": ["ulwrf", "rlu"],
                    "_rlucs_": ["ulwrf", "rlucs"],
                    "pfull": ["pres", "pres_layer"],
                    "phalf": ["pres", "pres_level"],
                    "_extra_3d_ta_": ["t", "temp_level", "param203.0.0", "tend_ta_mig"],
                      }
    variables_2d_sw = ["cosmu0","cosmu0_rt", "rsut", "rsds", "rvds_dir", "rvds_dif", 
                       "rpds_dir", "rpds_dif", "rnds_dir", "rnds_dif", "extra_2d_albedo", 
                       "extra_2d_iwp", "extra_2d_lwp", "extra_2d_prw", "clt", "albnirdir",
                       "albnirdif", "albvisdir", "albvisdif", "extra_2d_pr", "pr", "ps",
                       "pr_grpl", "pr_ice", "pr_rain", "pr_snow", "ts"]
    variables_2d_lw = ["ts_rad", "ts",  "rlut",  "extra_2d_albedo", "extra_2d_iwp", 
                       "extra_2d_lwp", "extra_2d_prw", "clt", "extra_2d_pr", "pr",
                       "pr_grpl", "pr_ice", "pr_rain", "pr_snow", "ps"]

    if parser.mode == "SW":
        with open("/p/project1/icon-a-ml/hafner1/cloudy_radiation/nn_config/SW_FLUX_with_HR.yml", "r") as file:
            args = yaml.safe_load(file)


        model = "/p/project1/icon-a-ml/hafner1/cloudy_radiation/trained_nns/SW_FLUX_HR_scripted_model_optimized_cpu.pt"
        if parser.partition == "train":    
            create_zarr(train_dates, variables_3d_sw, variables_2d_sw, sw=True, partition="train", model=model, args=args)
        elif parser.partition == "val":
            create_zarr(val_dates, variables_3d_sw, variables_2d_sw, sw=True, partition="val", model=model, args=args)
        elif parser.partition == "test": 
            create_zarr(test_dates, variables_3d_sw, variables_2d_sw, sw=True, partition="test", model=model, args=args)
        elif parser.partition == "all":
            create_zarr(train_dates, variables_3d_sw, variables_2d_sw, sw=True, partition="train", model=model, args=args)
            create_zarr(val_dates, variables_3d_sw, variables_2d_sw, sw=True, partition="val", model=model, args=args)
            create_zarr(test_dates, variables_3d_sw, variables_2d_sw, sw=True, partition="test", model=model, args=args)
        else:
            raise ValueError(f"Don't know {parser.partition}")
    elif parser.mode == "LW":
        with open("/p/project1/icon-a-ml/hafner1/cloudy_radiation/nn_config/LW_FLUX_with_HR.yml", "r") as file:
            args = yaml.safe_load(file)
        
        model = "/p/project1/icon-a-ml/hafner1/cloudy_radiation/trained_nns/LW_FLUX_HR_scripted_model_cpu.pt"

        if parser.partition == "train":
            create_zarr(train_dates, variables_3d_lw, variables_2d_lw, sw=False, partition="train", model=model, args=args)
        elif parser.partition == "val":
            create_zarr(val_dates, variables_3d_lw, variables_2d_lw, sw=False, partition="val", model=model, args=args)
        elif parser.partition == "test":
            create_zarr(test_dates, variables_3d_lw, variables_2d_lw, sw=False, partition="test", random_filter=False, model=model, args=args)
        elif  parser.partition == "all":
            create_zarr(train_dates, variables_3d_lw, variables_2d_lw, sw=False, partition="train", model=model, args=args)
            create_zarr(val_dates, variables_3d_lw, variables_2d_lw, sw=False, partition="val", model=model, args=args)
            create_zarr(test_dates, variables_3d_lw, variables_2d_lw, sw=False, partition="test", random_filter=False, model=model, args=args)
        else:
            raise ValueError(f"Don't know {parser.partition}")
    else:
        raise ValueError(f"Don't know {parser.mode}")

if __name__ == "__main__":
    main()