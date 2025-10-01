import numpy as np
from sklearn.metrics import r2_score

def get_random_rank3(ksamps, mask, plev, base_seed=1234):
    """
    Generate random numbers in [0, 1) only where mask is True.
    
    Parameters:
        ksamps (int): number of Monte Carlo samples
        mask (xr.DataArray or array-like): boolean mask (ncol, level)
        plev (xr.DataArray or array-like): pressure at interfaces (ncol, interface)
        base_seed (int): base seed value

    Returns:
        rank (np.ndarray): shape (ncol, level, ksamps), dtype=float64
    """
    # Extract numpy arrays if xarray
    mask_vals = mask.values if hasattr(mask, 'values') else np.array(mask)
    plev_vals = plev.values if hasattr(plev, 'values') else np.array(plev)

    kproma, n_interface = plev_vals.shape
    klev = n_interface - 1

    if mask_vals.shape != (kproma, klev):
        raise ValueError(f"mask must have shape (ncol, level) = ({kproma}, {klev})")

    rank = np.zeros((kproma, klev, ksamps), dtype=np.float64)
    p_surf_hpa = plev_vals[:, -1] / 100.0
    frac = p_surf_hpa - np.floor(p_surf_hpa)

    for jl in range(kproma):
        seed_val = int((frac[jl] * 1e8 + base_seed + jl) % (2**32))
        rng = np.random.default_rng(seed=seed_val)
        temp = rng.random((klev, ksamps)) 
        rank[jl] = np.where(mask_vals[jl, :, None], temp, 0.0)

    return rank


def apply_maxrand_overlap(rank, cld_frc_vals):
    """
    Apply maximum-random overlap: modifies rank in-place.

    Parameters:
        rank (np.ndarray): shape (ncol, level, ksamps), will be updated
        cld_frc_vals (np.ndarray): cloud fraction, shape (ncol, level)

    Returns:
        is_cloudy (np.ndarray): bool array (ncol, level, ksamps)
    """
    kproma, klev, ksamps = rank.shape
    one_minus = 1.0 - cld_frc_vals

    for jk in range(1, klev):
        for jl in range(kproma):
            for js in range(ksamps):
                if rank[jl, jk-1, js] > one_minus[jl, jk-1]:
                    rank[jl, jk, js] = rank[jl, jk-1, js]
                else:
                    rank[jl, jk, js] *= one_minus[jl, jk-1]

    return rank > one_minus[:, :, None]


def sample_cloud_state(ngpt, plev, cld_frc):
    """
    Sample cloud state using maximum-random overlap.

    Parameters:
        ngpt (int): number of Monte Carlo samples (ksamps)
        plev (xr.DataArray or array-like): pressure at interfaces, shape (ncol, interface)
        cld_frc (xr.DataArray or array-like): cloud fraction, shape (ncol, level)

    Returns:
        cloud_mask (np.ndarray): boolean array of shape (ncol, level, ngpt)
    """
    # Extract values
    cld_frc_vals = cld_frc.values if hasattr(cld_frc, 'values') else np.array(cld_frc)
    plev_vals = plev.values if hasattr(plev, 'values') else np.array(plev)

    kproma, klev = cld_frc_vals.shape

    if plev_vals.shape != (kproma, klev + 1):
        raise ValueError(f"plev must have shape (ncol, level+1) = ({kproma}, {klev+1})")

    # Create mask: where cloud exists
    mask = cld_frc_vals > 0.0

    # Step 1: Generate randoms
    rank = get_random_rank3(ksamps=ngpt, mask=mask, plev=plev_vals, base_seed=1234)

    # Step 2: Apply max-ran overlap
    cloud_mask = apply_maxrand_overlap(rank, cld_frc_vals)

    return cloud_mask  # shape: (ncol, level, ngpt), dtype=bool, plain NumPy

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
    rel_x_pred = np.divide( x, np.abs(y_true), where = np.abs(y_true)>0)
    rel_x_true = np.divide( x, np.abs(y_pred), where = np.abs(y_pred)>0)


    return {"mae": mae,
            "rel_mae_true": np.mean(rel_x_true, axis=0),
            "rel_mae_pred": np.mean(rel_x_pred, axis=0),
            "rel_true_perc_5": np.percentile(rel_x_true, 2.5, axis=0),
            "rel_pred_perc_5": np.percentile(rel_x_pred, 2.5, axis=0),
            "rel_true_perc_95": np.percentile(rel_x_true, 97.5, axis=0),
            "rel_pred_perc_95": np.percentile(rel_x_pred, 97.5, axis=0),
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

def print_latex_table(label, stats):
    mae = f'{np.mean(stats["mae"]):.3f} ({np.mean(stats["mae"]/np.abs(stats["true"]))*100:.2f} \\%)'
    rmse = f'{np.mean(stats["rmse"]):.3f} ({np.mean(stats["rmse"]/np.abs(stats["true"]))*100:.2f} \\%)'
    bias = f'{np.mean(stats["mean"]):.3f} ({np.mean(stats["mean"]/np.abs(stats["true"]))*100:.2f} \\%)'
    print(f'{label} & {mae} & {bias} & {stats["r2_mean"]:.2f} & {rmse} \\\\')
