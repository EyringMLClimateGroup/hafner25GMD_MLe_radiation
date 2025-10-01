import pyrte_rrtmgp
import xarray as xr 
from pyrte_rrtmgp import rrtmgp_cloud_optics, rrtmgp_gas_optics
from pyrte_rrtmgp.rte_solver import rte_solve
from pyrte_rrtmgp.data_types import (
    CloudOpticsFiles,
    GasOpticsFiles,
    OpticsProblemTypes,
)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys
import os


from utils.mcica import sample_cloud_state

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default='SW', choices=['SW', 'LW'])
parser.add_argument("-s", "--set", default="april", choices=["april", "november", "january", "july"])
parser = parser.parse_args()
print(f"Calculating heating rates for: ", parser.mode, parser.set) 

mcica = True
mcica_snow = False

# Load data 
da_all = xr.open_dataset(f"/p/project1/icon-a-ml/hafner1/cloudy_radiation/data/{parser.set}/test_{parser.mode}.zarr", engine="zarr")
da = da_all #.isel({"time": [0,1,2], "cell": np.arange(4000)}) # this is only for testing
ghg = xr.open_mfdataset("/p/project1/icon-a-ml/hafner1/cloudy_radiation/pyrte_rrtmgp_test/greenhouse_historical_plus.nc", engine="netcdf4")
zh_lr_path = "/p/project1/icon-a-ml/hafner1/org_radiation/preprocessing/atm_amip_R2B5_vgrid_ml.nc"
zh_lr_ds = xr.open_mfdataset(zh_lr_path)
zh_lr = zh_lr_ds["zhalf"].values
dz_lr = zh_lr_ds["dzghalf"].rename({"ncells": "cell"})
height = np.mean(zh_lr_ds["zfull"].values, axis=-1)/1000
vweights = np.load("/p/project1/icon-a-ml/hafner1/cloudy_radiation/preprocessing/weights_vert_integral_on_cg.npy")
result_folder = "/p/project1/icon-a-ml/hafner1/cloudy_radiation/results/pyrte_snow/"
os.makedirs(result_folder, exist_ok=True)


for f in ghg.variables:
    if f in ["time", "lon", "lat"]:
        continue
    if parser.set == "november":
        year = 2004
    else:
        year = 2005
    da[f.lower()] = ghg[f][year].values.squeeze()* float(ghg[f].units)    
    print(f.lower(), ghg[f][year].values.squeeze()* float(ghg[f].units))
da["co"] = 0
da["o2"] = 0.2095
da["n2"] = 0.7808

# Load gasoptics files    
if parser.mode == "LW":
    cloud_optics = rrtmgp_cloud_optics.load_cloud_optics(
        cloud_optics_file=CloudOpticsFiles.LW_BND 
    )
    gas_optics = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256 
    )
elif parser.mode == "SW":
    cloud_optics = rrtmgp_cloud_optics.load_cloud_optics(
        cloud_optics_file=CloudOpticsFiles.SW_BND 
    )
    gas_optics = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.SW_G112 
    )

def content_to_path(content, rho, zhalf):
    # units: kg/kg to g/m^2 
    dz = (zhalf.isel({"height_2": np.arange(0,47)}).values - zhalf.isel({"height_2": np.arange(1,48)}).values)
    if dz.shape[0] == 47:
        dz = np.transpose(dz, (2,0,1))
    return content * rho * dz * 1000
    
#sel_dz = dz_lr.sel({"cell": da["cell_idx"].astype("int").values.squeeze() })
da["lwp"] = content_to_path(da["extra_3d_clw"], da["rho"], da["zhalf"])
da["iwp"] = content_to_path(da["extra_3d_cli"], da["rho"], da["zhalf"])
da["swp"] = content_to_path(da["extra_3d_qs"], da["rho"], da["zhalf"])

# Get reference radii values
rel_val = 0.5 * (cloud_optics["radliq_lwr"] + cloud_optics["radliq_upr"])
rei_val = 0.5 * (cloud_optics["diamice_lwr"] + cloud_optics["diamice_upr"])/2

da["rel"] = xr.where(da.lwp > 0., rel_val, 0)  
da["rei"] = xr.where(da.iwp > 0,  rei_val, 0)  

rename_dict = {"cfc_12": "cfc12",
              "cfc_11": "cfc11",
              "ts": "surface_temperature",
              "cell": "column",
              "extra_3d_hus": "h2o",
              "height_2": "level",
              "height": "layer",
              "extra_3d_ta": "temp_layer",
              }
atmosphere =  da.rename(rename_dict)

# Clip values to avoid nans in output
atmosphere["temp_layer"] = xr.ufuncs.maximum(
                            atmosphere.temp_layer,
                            gas_optics.compute_gas_optics.temp_min + sys.float_info.epsilon, 
                    )
atmosphere["pres_layer"] = xr.ufuncs.maximum(
                            atmosphere.pres_layer,
                            gas_optics.compute_gas_optics.press_min + sys.float_info.epsilon, 
                    )

if parser.mode == "LW":
    print(f"Calculating gas optics and clear sky fluxes...")
    optical_props_list = []
    fluxes_clear_list = []
    for t in tqdm(range(len(atmosphere.time))):
        this_atmosphere = atmosphere.isel({"time": t})
        optical_props = gas_optics.compute_gas_optics(
            this_atmosphere.drop_vars(["layer", "level"]) ,
            problem_type=OpticsProblemTypes.ABSORPTION, 
            add_to_input=False
        )
        optical_props["surface_emissivity"] = 0.996 # Value used in ICON
        optical_props_list.append(optical_props)
        #fluxes_clear = rte_solve(optical_props, add_to_input=False)
        #fluxes_clear_list.append(fluxes_clear)
        
    #print(f"Saving clear sky fluxes for {parser.mode}...")
    #xr.merge([f.expand_dims(dim="time") for f in fluxes_clear_list]
    #        ).to_zarr(f"{result_folder}{parser.mode}_{parser.set}_clear.zarr", mode="w")

    print(f"Adding cloud optics and calculating cloudy-sky fluxes...")
    fluxes_cloudy_list = []
    for t in tqdm(range(len(atmosphere.time))):
        this_atmosphere = atmosphere.isel({"time": t})
        cloud_optical_props = cloud_optics.compute_cloud_optics(
            this_atmosphere,
            problem_type=OpticsProblemTypes.ABSORPTION,
            add_to_input = False
        )
        if mcica:
            cloud_mask = sample_cloud_state(len(cloud_optical_props.bnd), 
                                          this_atmosphere["pres_level"].T,
                                          this_atmosphere["cl"].T)
            
            cloud_mask_xr = xr.DataArray(
                cloud_mask,
                dims=["column", "layer", "bnd"],
                name="mask"
            )
            
            cloud_optical_props["tau"] = xr.where(
                cloud_mask_xr,
                cloud_optical_props["tau"],
                0
            )
        # changes optical_props
        cloud_optical_props.add_to(optical_props_list[t])

        #fluxes_cloudy = rte_solve(optical_props_list[t], add_to_input=False)
        #fluxes_cloudy_list.append(fluxes_cloudy)

    #xr.merge([f.expand_dims(dim="time") for f in fluxes_clear_list]
    #        ).to_zarr(f"{result_folder}{parser.mode}_{parser.set}_cloudy.zarr", mode="w")
    
    print(f"Adding snow optics and calculating all-sky fluxes...")
    fluxes_all_list = []
    for t in tqdm(range(len(atmosphere.time))):
        this_atmosphere = atmosphere.isel({"time": t})
        this_atmosphere["rel"] = this_atmosphere["rei"]
        this_atmosphere["lwp"] = xr.zeros_like(this_atmosphere["lwp"])
        this_atmosphere["iwp"] = this_atmosphere["swp"]
        
        snow_optical_props = cloud_optics.compute_cloud_optics(
            this_atmosphere,
            problem_type=OpticsProblemTypes.ABSORPTION,
            add_to_input = False
        )
        if mcica_snow:
            snow_mask = sample_cloud_state(len(snow_optical_props.bnd), 
                                          this_atmosphere["pres_level"].T,
                                          this_atmosphere["cl"].T)
            
            snow_mask_xr = xr.DataArray(
                snow_mask,
                dims=["column", "layer", "bnd"],
                name="mask"
            )
            
            snow_optical_props["tau"] = xr.where(
                snow_mask_xr,
                snow_optical_props["tau"],
                0
            )
        # changes optical_props
        snow_optical_props.add_to(optical_props_list[t])

        fluxes_all = rte_solve(optical_props_list[t], add_to_input=False)
        fluxes_all_list.append(fluxes_all)
    print(f"Saving all sky fluxes for {parser.mode}...")
    xr.merge([f.expand_dims(dim="time") for f in fluxes_all_list]
            ).to_zarr(f"{result_folder}{parser.mode}_{parser.set}_all.zarr", mode="w")
    
elif parser.mode == "SW":
    print(f"Calculating gas optics and clear sky fluxes...")
    optical_props_list = []
    fluxes_clear_list = []
    atmosphere["total_solar_irradiance"] = atmosphere["toa"]/atmosphere["cosmu0"]
    for t in tqdm(range(len(atmosphere.time))):
        this_atmosphere = atmosphere.isel({"time": t})
        optical_props = gas_optics.compute_gas_optics(
            this_atmosphere.drop_vars(["layer", "level"]) ,
            problem_type=OpticsProblemTypes.TWO_STREAM, 
            add_to_input=False
        )
        optical_props["mu0"] = this_atmosphere["cosmu0"]
        optical_props["surface_albedo"] = this_atmosphere["extra_2d_albedo"]
        optical_props_list.append(optical_props)
        #fluxes_clear = rte_solve(optical_props, add_to_input=False)
        #fluxes_clear_list.append(fluxes_clear)
        
    #print(f"Saving all sky fluxes for {parser.mode}...")
    #xr.merge([f.expand_dims(dim="time") for f in fluxes_clear_list]
    #        ).to_zarr(f"{result_folder}{parser.mode}_{parser.set}_fluxes_clear.zarr", mode="w")

    print(f"Adding cloud optics and calculating cloudy-sky fluxes...")
    fluxes_cloudy_list = []
    for t in tqdm(range(len(atmosphere.time))):
        this_atmosphere = atmosphere.isel({"time": t})
        cloud_optical_props = cloud_optics.compute_cloud_optics(
            this_atmosphere,
            problem_type=OpticsProblemTypes.TWO_STREAM,
            add_to_input = False
        )
        if mcica:
            cloud_mask = sample_cloud_state(len(cloud_optical_props.bnd), 
                                          this_atmosphere["pres_level"].T,
                                          this_atmosphere["cl"].T)
            
            cloud_mask_xr = xr.DataArray(
                cloud_mask,
                dims=["column", "layer", "bnd"],
                name="mask"
            )
            
            cloud_optical_props["tau"] = xr.where(
                cloud_mask_xr,
                cloud_optical_props["tau"],
                0
            )
            cloud_optical_props["ssa"] = xr.where(
                cloud_mask_xr,
                cloud_optical_props["ssa"],
                1
            )
            cloud_optical_props["g"] = xr.where(
                cloud_mask_xr,
                cloud_optical_props["g"],
                0
            )
        # changes optical_props
        cloud_optical_props.add_to(optical_props_list[t])

        #fluxes_cloudy = rte_solve(optical_props_list[t], add_to_input=False)
        #fluxes_cloudy_list.append(fluxes_cloudy)

    #xr.merge([f.expand_dims(dim="time") for f in fluxes_cloudy_list]
    #        ).to_zarr(f"{result_folder}{parser.mode}_{parser.set}_fluxes_cloudy.zarr", mode="w")
    
    print(f"Adding snow optics and calculating all-sky fluxes...")
    fluxes_all_list = []
    for t in tqdm(range(len(atmosphere.time))):
        this_atmosphere = atmosphere.isel({"time": t})
        this_atmosphere["rel"] = this_atmosphere["rei"]
        this_atmosphere["lwp"] = xr.zeros_like(this_atmosphere["lwp"])
        this_atmosphere["iwp"] = this_atmosphere["swp"]
        
        snow_optical_props = cloud_optics.compute_cloud_optics(
            this_atmosphere,#.drop_vars(["layer", "level"]).compute() , 
            problem_type=OpticsProblemTypes.TWO_STREAM,
            add_to_input = False
        )
        if mcica_snow:
            snow_mask = sample_cloud_state(len(snow_optical_props.bnd), 
                                          this_atmosphere["pres_level"].T,
                                          this_atmosphere["cl"].T)
            
            snow_mask_xr = xr.DataArray(
                snow_mask,
                dims=["column", "layer", "bnd"],
                name="mask"
            )
            
            snow_optical_props["tau"] = xr.where(
                snow_mask_xr,
                snow_optical_props["tau"],
                0
            )
            snow_optical_props["ssa"] = xr.where(
                snow_mask_xr,
                snow_optical_props["ssa"],
                1
            )
            snow_optical_props["g"] = xr.where(
                snow_mask_xr,
                snow_optical_props["g"],
                0
            )
        # changes optical_props
        snow_optical_props.add_to(optical_props_list[t])

        fluxes_all = rte_solve(optical_props_list[t], add_to_input=False)
        fluxes_all_list.append(fluxes_all)
    print(f"Saving all sky fluxes for {parser.mode}...")
    xr.merge([f.expand_dims(dim="time") for f in fluxes_all_list]
            ).to_zarr(f"{result_folder}{parser.mode}_{parser.set}_all.zarr", mode="w")
    
    