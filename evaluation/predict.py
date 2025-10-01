import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

def cloudy_predict(model, dataset, variables, model_type):
    """
    Predicts the output using the given model and data generator.
    Parameters:
    - model: The trained model used for prediction.
    - data_gen: The data generator object that generates input data for prediction.
    - variables: A dictionary containing the input and output variables.
    - model_type: The type of the model used for prediction.
    Returns:
    - return_dict: A dictionary containing the predicted and true values for each variable.
    """
    if "SW" in model_type:
        hr_cre = "tend_ta_rsw_cre" 
        hr_cs = "tend_ta_rsw_cs" 
        hr    = "tend_ta_rsw" 
        ds    = "rsds" # downward flux surface
        ut    = "rsut" # upward flux toa
        dn_flx_cs = "rsdcs"
        up_flx_cs = "rsucs"
    elif "LW" in model_type:
        hr_cre = "tend_ta_rlw_cre" 
        hr_cs = "tend_ta_rlw_cs" 
        hr    = "tend_ta_rlw" 
        ds="rlds_rld" # downward flux surface
        ut="rlut" # upward flux toa
        dn_flx_cs = "rldcs"
        up_flx_cs = "rlucs"
    
    return_dict = {}
    
    for v in dataset.ds:
        s = dataset.ds[v].isel({"time": 0}).shape
        if len(s) == 2:
            continue
        return_dict[v] = []
    return_dict["t"] = []
    return_dict["t_i"] = []
    return_dict[f"pred_{hr}"] = []
    return_dict[f"true_{hr}"] = []
    return_dict[f"true_{hr_cre}"] = []
    return_dict[f"pred_{hr_cre}"] = []
    return_dict[f"true_{ds}"] = []
    return_dict[f"true_{ut}"] = []
    return_dict["cvair"] = []
    return_dict["cl"] = []
    if "FLUX_HR" in model_type:
        return_dict[f"pred_{ds}"] = []
        return_dict[f"pred_{ut}"] = []
        if "SW" in model_type:
            for v in variables["out_vars"][3:]:                
                return_dict[f"true_{v}"] = []
                return_dict[f"pred_{v}"] = []
    
    
    print(return_dict.keys())
    for i in tqdm(range(0, len(dataset.ds.time))): #len(dataset.ds.time)
        this_ds = dataset.ds.isel({"time": i})
        return_dict["t"].append(np.array([this_ds.time.values]*len(this_ds.cell)))
        return_dict["t_i"].append(np.array([i]*len(this_ds.cell)))
        for v in dataset.ds:
            s = this_ds[v].shape
            if len(s) == 2:
                if v in ["cvair", "cl"]:
                    return_dict[v].append(this_ds[v].values.T)
                continue
            else:
                return_dict[v].append(this_ds[v].values)
            
        x, y = dataset.prepare_for_testing(this_ds)
        y_pred =  model.predict(x).detach().numpy()
        
        return_dict[f"pred_{hr}"].append( y_pred[:,:47] + this_ds[hr_cs].values.T)
        return_dict[f"true_{hr}"].append( this_ds[hr].values.T)
        return_dict[f"pred_{hr_cre}"].append( y_pred[:,:47])
        return_dict[f"true_{hr_cre}"].append( (this_ds[hr]-this_ds[hr_cs]).values.T )
        return_dict[f"true_{ds}"].append( this_ds[ds].values)        
        return_dict[f"true_{ut}"].append( this_ds[ut].values )
        if "FLUX_HR" in model_type:
            dscs = this_ds[dn_flx_cs][-1].values
            utcs = this_ds[up_flx_cs][0].values
            if "SW" in model_type:
                scaling = (this_ds["cosmu0_rt"]/this_ds["cosmu0"]).values
                return_dict[f"pred_{ds}"].append( y_pred[:,47] + dscs*scaling )        
                return_dict[f"pred_{ut}"].append( y_pred[:,47+1] + utcs*scaling )
            
                for idx, v in enumerate(variables["out_vars"][3:]):
                    return_dict[f"pred_{v}"].append( y_pred[:,47+idx+2])
                    return_dict[f"true_{v}"].append( this_ds[v].values)
            else:
                return_dict[f"pred_{ds}"].append( y_pred[:,47] + dscs )        
                return_dict[f"pred_{ut}"].append( y_pred[:,47+1] + utcs )
            
    for v in return_dict.keys():
        if len(return_dict[v][0].shape) == 2:
            return_dict[v] = np.vstack(return_dict[v])
        else:
            return_dict[v] = np.concatenate(return_dict[v])
    return_dict["cell_idx"] = return_dict["cell_idx"].astype("int")
    return return_dict


    
def statistics(y_true, y_pred):
    """
    Calculate various statistical metrics to evaluate the performance of a prediction model.
    Parameters:
    - y_true (array-like): True values of the target variable.
    - y_pred (array-like): Predicted values of the target variable.
    Returns:
    - dict: A dictionary containing statistical metrics such as mean absolute error, mean squared error, and R^2 score.
    """

    x = np.abs(y_true - y_pred)
    mae = np.mean(x, axis=0)
    std_m = np.std(x, axis=0, where=x<mae)
    std_p = np.std(x, axis=0, where=x>mae)
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    rel_x_pred = np.where(np.abs(y_true)>0, x/np.abs(y_true), 0)
    rel_x_true = np.where(np.abs(y_pred)>0, x/np.abs(y_pred), 0)


    return {"mae": mae,
            "rel_mae_true": np.mean(rel_x_true, axis=0),
            "rel_mae_pred": np.mean(rel_x_pred, axis=0),
            "rel_true_perc_5": np.percentile(rel_x_true, 5, axis=0),
            "rel_pred_perc_5": np.percentile(rel_x_pred, 5, axis=0),
            "rel_true_perc_95": np.percentile(rel_x_true, 95, axis=0),
            "rel_pred_perc_95": np.percentile(rel_x_pred, 95, axis=0),
            "true": np.mean(y_true, axis=0),
            "pred": np.mean(y_pred, axis=0),
            "mean": np.mean(y_true - y_pred, axis=0),
            "mean_err": np.mean(y_true - y_pred, axis=0),
            "mse": np.mean(x**2, axis=0),
            "rmse": np.sqrt(np.mean(x**2, axis=0)),
            "abs_median": np.median(x, axis=0),
            "median": np.median(y_true - y_pred, axis=0),
            "std_p": std_p,
            "std_m": std_m,
            "min": np.min(y_true - y_pred, axis=0),
            "max": np.max(y_true - y_pred, axis=0),
            "max_abs": np.max(x, axis=0),
            "abs_perc_5": np.percentile(x, 5, axis=0),
            "abs_perc_95": np.percentile(x, 95, axis=0),
            "perc_5": np.percentile(y_true - y_pred, 5, axis=0),
            "perc_95": np.percentile(y_true - y_pred, 95, axis=0),
            "r2": r2,
            "r2_mean": np.mean(r2),
           }


def summary_statistics(result_dict, grid, model_type, varlist, verbose=False):
    """
    Calculate summary statistics for a given result dictionary.
    Parameters:
    - result_dict (dict): A dictionary containing the results.
    - grid (dict): A dictionary containing the grid information.
    - model_type (str): The type of the model.
    - varlist (list): A list of variables for which the variables will be calculated.
    - verbose (bool, optional): Whether to print verbose output. Defaults to False.
    Returns:
    - summary_dict (dict): A dictionary containing the summary statistics.
    """

    clat = grid["clat"]
    clon = grid["clon"]
    n = len(clat)

    summary_dict = {
        "clat": clat,
        "clon": clon}
    print(result_dict.keys())

    for v in varlist:
        if "tend" in v:
            summary_dict[f"mae_{v}"] = np.zeros((n, 47))
            summary_dict[f"bias_{v}"] = np.zeros((n, 47))
            summary_dict[f"r2_{v}"] = np.zeros((n, 47))
            summary_dict[f"true_{v}"] = np.zeros((n, 47))
            summary_dict[f"pred_{v}"] = np.zeros((n, 47))
            summary_dict[f"rel_{v}"] = np.zeros((n, 47))
        else:
            summary_dict[f"mae_{v}"] = np.zeros((n, 1))
            summary_dict[f"bias_{v}"] = np.zeros((n, 1))
            summary_dict[f"r2_{v}"] = np.zeros((n, 1))
            summary_dict[f"true_{v}"] = np.zeros((n, 1))
            summary_dict[f"pred_{v}"] = np.zeros((n, 1))
            summary_dict[f"rel_{v}"] = np.zeros((n, 1))
    f = 0
    for i in tqdm(range(n)):
        idx = np.argwhere(result_dict["cell_idx"]==i).squeeze()
        
        
        for v in varlist:
            if idx.size == 0:
                if verbose:
                    print("Failed to produce summary for index ", i)
                f += 1
                continue
            summary_dict[f"mae_{v}"][i] = np.mean(np.abs(result_dict[f"true_{v}"][idx] - result_dict[f"pred_{v}"][idx]), axis=0)
            summary_dict[f"bias_{v}"][i] = np.mean((result_dict[f"true_{v}"][idx] - result_dict[f"pred_{v}"][idx]), axis=0)
            summary_dict[f"r2_{v}"][i] = r2_score(result_dict[f"true_{v}"][idx], result_dict[f"pred_{v}"][idx], multioutput="raw_values")
            summary_dict[f"true_{v}"][i] = np.mean(result_dict[f"true_{v}"][idx], axis=0)
            summary_dict[f"pred_{v}"][i] = np.mean(result_dict[f"pred_{v}"][idx], axis=0)
            summary_dict[f"rel_{v}"][i] = np.mean(np.abs(np.divide(result_dict[f"true_{v}"][idx] - result_dict[f"pred_{v}"][idx], np.abs(result_dict[f"true_{v}"][idx]), where=np.abs(result_dict[f"true_{v}"][idx])>1e-2)), axis=0)
    
    print(f"Failed to produce summary for {f} indices.")
    return summary_dict

def summary_statistics_var(result_dict, vname, grid):
    """
    Calculate the summary statistics for a given variable in a result dictionary.
    Parameters:
    - result_dict (dict): A dictionary containing the results.
    - vname (str): The name of the variable to calculate the summary statistics for.
    - grid (dict): A dictionary containing the grid information.
    Returns:
    - summary_dict (dict): A dictionary containing the summary statistics.
        - "clat" (array-like): The latitude values of the grid.
        - "clon" (array-like): The longitude values of the grid.
        - f"mean_{vname}" (array-like): The mean values of the variable for each grid point.
    """

    clat = grid["clat"]
    clon = grid["clon"]
    n = len(clat)
    var = result_dict[vname]
    s = var.shape[-1] if len(var.shape) == 2 else 1

    summary_dict = {
        "clat": clat,
        "clon": clon,
        f"mean_{vname}": np.zeros((n, s)),
    }
    for i in tqdm(range(n)):
        idx = np.argwhere(result_dict["cell_idx"]==i).squeeze()
        summary_dict[f"mean_{vname}"][i] = np.mean(var[idx], axis=0)

    return summary_dict