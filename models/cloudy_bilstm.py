
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from .preprocessing_layer import Preprocessing
from .postprocessing_layer import Postprocessing
import numpy as np

class Cloudy_BiLSTM(L.LightningModule):
    def __init__(self, model_type, output_features, norm_file, in_vars, out_vars, extra_shape=0, hidden_size=96, n_layer=1, var_len = 47, lr=1.e-3, weight_decay=0):
        super(Cloudy_BiLSTM, self).__init__()
        self.model_type = model_type
        self.extra_shape = extra_shape
        self.var_len = var_len
        self.in_vars = in_vars
        self.lr = lr
        self.weight_decay = weight_decay
        input_features = len(in_vars)
        self.lstm_output = torch.Tensor()
        self.prep = Preprocessing(norm_file, in_vars, mode="horizontal", pad_len=47, var_len=self.var_len)
        self.post = Postprocessing(out_vars, in_vars)
        self.lstm = nn.LSTM(input_features, hidden_size, bidirectional=True, batch_first=True, num_layers = n_layer)

        self.linear = nn.Linear(hidden_size*2, output_features)
        self.relu = nn.ReLU()

    def forward(self, full_input):
        if self.extra_shape > 0:
            x_in = full_input[:,  :-self.extra_shape]
        else:
            x_in = full_input
            
        x_norm = self.prep(x_in)
        x, _ = self.lstm(x_norm)
        self.lstm_output = x
        output = self.linear(x).squeeze()    
        denormalized_output = self.post(output, full_input)

        return denormalized_output

    def predict(self, x):
        self.eval()
        y_hat = self(x)
        return y_hat
 
    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch, v="val_")
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        return loss 
    
    def CRPS_loss(self, batch):
        x, tend_ta = batch[0].float(), batch[1].float()
        e = x[:, -self.extra_shape: ]
        l = self.var_len
        tend_ta_cs = e[:, :l]
        y = (tend_ta - tend_ta_cs)*86400 # K/d
        n_ensembles = 10
        noise_size = 0.05
    
        y_hat_ensembles = torch.zeros((n_ensembles, *y.shape)).to(x)
        for i in range(n_ensembles):
            noise = noise_size*torch.rand(x.shape)+1-noise_size/2
            noise = noise.to(x)
            y_hat_ensembles[i] = self(x*noise)*86400 # K/d
       
        y_hat_mean = torch.mean(y_hat_ensembles, dim=0)
        y_hat_asc = torch.sort(y_hat_ensembles, dim=0)[0]
        y_hat_des = torch.sort(y_hat_ensembles, dim=0,descending=True)[0]
        
        MAE_term = torch.mean(torch.abs(y_hat_mean-y), dim=0)
        spread_term = torch.mean(0.5*torch.mean(torch.abs(y_hat_asc - y_hat_des), dim=0), dim=0)
        CRPS_loss=torch.abs(MAE_term-spread_term)
        return torch.mean(CRPS_loss)
                
    def loss(self, batch, v=""):
        x, tend_ta =  batch[0].float(), batch[1].float()
        y_hat = self(x)
        e = x[:, -self.extra_shape: ]
        l = self.var_len
        tend_ta_cs = e[:, :l]
        y = tend_ta - tend_ta_cs
        q = e[:, l:2*l]
        cvair = e[:, 2*l:3*l]
        vweights = e[:,3*l:4*l]
        q_r = (y_hat+tend_ta_cs)*cvair*vweights # hr_hat in K/s because qconv converts K/s to W/m^2
        fnet_hr = torch.sum(q_r, dim=-1) #weights for vertical integral correction due to coarse graining 
        
        # MAE and MSE in K/d and (K/d)**2 for comparison of other trainings
        scaling_factor = torch.ones_like(y)*86400
        
        mae = F.l1_loss(y_hat*scaling_factor, y*scaling_factor)
        mse = F.mse_loss(y_hat*scaling_factor, y*scaling_factor)
        q_loss = torch.mean(torch.square(torch.mul((y_hat-y)*scaling_factor,q)))
        n = 4 if "extra_3d_qs" in self.in_vars else 3
        if "SW" in self.model_type:
            rsdt = x[:,l*n]  # toa
            alb = x[:,l*n+1] # albedo
            rsds = e[:,-2]
            rsut = e[:,-1]
            fnet_flux = ((rsdt-rsut)-(1-alb)*rsds) # denormalized net-flux
        elif "LW" in self.model_type:
            rlds = e[:,-2]
            rlut = e[:,-1]
            ts_rad = x[:,n*l]
            sig = 5.670374419e-8
            em = 0.996
            rlus = em*sig*ts_rad**4
            fnet_flux = (-rlut-(rlds-rlus)) + 1.41 # denormalized net-flux
            # constant -1.41 is the bias that is also in the dataset
            # due to correction by surface scheme 
        energy =  torch.mean(torch.abs(fnet_flux - fnet_hr))
        # not adding energy term here. only add it when training for fluxes
        loss = mae + mse
        self.log(f'{v}mae_loss', mae, sync_dist=True, prog_bar=True)
        self.log(f'{v}mse_mae_loss', mse + mae, sync_dist=True)
        self.log(f'{v}mse_loss', mse, sync_dist=True, prog_bar=True)      
        self.log(f'{v}energy_loss', energy, sync_dist=True, prog_bar=True)
        self.log(f'{v}q_loss', q_loss, sync_dist=True)
        return loss

    def on_after_backward(self):
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:  # log every n steps
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params.grad, self.trainer.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=0.001, threshold_mode='rel')
        return {"optimizer":optimizer, "lr_scheduler":{"scheduler": scheduler, "monitor": "val_mse_mae_loss"}}


class Cloudy_BiLSTM_with_Flux(L.LightningModule):
    def __init__(self, hr_model, model_type, output_features, in_vars, out_vars, extra_shape=0, hidden_size=96, lr=1.e-3, weight_decay=0.0, shap=False):
        super(Cloudy_BiLSTM_with_Flux, self).__init__()
        self.model_type = model_type
        self.extra_shape = extra_shape
        self.output_len = output_features
        self.lr = lr
        self.weight_decay = weight_decay
        self.hr_model = hr_model
        self.shap = shap
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.post = Postprocessing(out_vars[1:], in_vars)
        l = 47
        if "SW" in model_type:
            self.n_e = 4 # for the partial albedos
        else:
            self.n_e = 0  
        self.linear = nn.Sequential(nn.Linear(hidden_size*2, 1), nn.Tanh())
        n_hidden = 32
        self.out = nn.Sequential(nn.Linear(l+self.n_e, n_hidden), nn.Tanh(), nn.Linear(n_hidden, output_features), nn.Tanh())
        
    def forward(self, full_input):
        if self.extra_shape > 0:
            x, q = full_input[:,  :-self.extra_shape], full_input[:,  -self.extra_shape:]
        else:
            x = full_input
            q = torch.zeros_like(x)
        if "SW" in self.model_type:
            n_e = self.n_e
            e = x[:,-n_e:]
            x = x[:,:-n_e]
        else:
            e = torch.zeros_like(x)
        
        if self.shap:
            hr = self.hr_model(x)
            lstm_output = self.hr_model.lstm_output
            
        else:
            with torch.no_grad():
                hr = self.hr_model(x)
                lstm_output = self.hr_model.lstm_output
        feat = self.linear(lstm_output).squeeze()
        if "SW" in self.model_type:
            flux_feat = torch.cat((feat, e), dim=-1)  
        else:
            flux_feat = feat
        output = self.post(self.out(flux_feat), x)
        output = torch.cat((hr, output), dim=-1) # return only hr and flux
        return output

    def predict(self, x):
        self.eval()
        y_hat = self(x)
        return y_hat 
    
 
    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch, v="val_")
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        return loss 

    def loss(self, batch, v=""):
        x, y = batch[0].float(), batch[1].float()
        y_hat = self(x)
        hr_hat, flux_cri_hat = y_hat[:,:-self.output_len], y_hat[:,-self.output_len:]
        tend_ta = y[:,:-self.output_len]
        flux_all = y[:,-self.output_len:]
        e = x[:, -self.extra_shape: ]
        l = self.hr_model.var_len

        # Loss hr as above
        tend_ta_cs = e[:, :l]
        y_cri = tend_ta - tend_ta_cs
        q = e[:, l:2*l]
        cvair = e[:, 2*l:3*l]
        vweights = e[:,3*l:4*l]
        q_r = (hr_hat+tend_ta_cs)*cvair*vweights # hr_hat in K/s because qconv converts K/s to W/m^2
        fnet_hr = torch.sum(q_r, dim=-1) #weights for vertical integral correction due to coarse graining 
        
        # MAE and MSE in K/d and (K/d)**2 for comparison of other trainings
        scaling_factor = torch.ones_like(y_cri)*86400
        mae_hr = F.l1_loss(hr_hat*scaling_factor, y_cri*scaling_factor)
        mse_hr = F.mse_loss(hr_hat*scaling_factor, y_cri*scaling_factor)
        q_loss_hr = torch.mean(torch.square(torch.mul((hr_hat-y_cri)*scaling_factor,q)))
        
        #self.log(f'{v}hr_mae_loss', mae_hr, sync_dist=True, prog_bar=True)
        #self.log(f'{v}hr_mse_mae_loss', mse_hr + mae_hr, sync_dist=True)
        #self.log(f'{v}hr_mse_loss', mse_hr, sync_dist=True, prog_bar=True)      
        #self.log(f'{v}q_loss', q_loss_hr, sync_dist=True)
        # --------------------
        
        # Flux loss
        flux_cri = flux_all.clone() # SW fluxes include partial fluxes that are predicted as they are 
        dscs = e[:,-2] # downward surface clear sky flux
        utcs = e[:,-1] # upward flux at TOA
        flux_cri[:,0] -= dscs # subtract clear sky flux for downward surface to get cloud effect
        flux_cri[:,1] -= utcs # subtract clear sky flux for upward TOA to get cloud effect
    
        mse = F.mse_loss(flux_cri_hat, flux_cri)
        mae = F.l1_loss(flux_cri_hat, flux_cri)
        mbe = torch.abs(torch.mean(flux_cri_hat - flux_cri)) # mean bias error
        self.log(f'{v}mse_loss', mse, sync_dist=True)
        self.log(f'{v}mae_loss', mae, sync_dist=True)
        self.log(f'{v}mse_mae_loss', mse + mae, sync_dist=True)
        self.log(f'{v}mbe', mbe, sync_dist=True)
        # ----------------------

        # calculate additional loss terms like energy and direct/diffuse
        n = 4 if "extra_3d_qs" in self.in_vars else 3
        if "SW" in self.model_type:
            rsdt = x[:,l*n]  # toa
            alb = x[:,l*n+1] # albedo
            rsds = flux_cri_hat[:,0] + dscs # all aky flux
            rsut = flux_cri_hat[:,1] + utcs # all sky flux
            fnet_flux = ((rsdt-rsut)-(1-alb)*rsds) # denormalized net-flux

            rvds_dir = flux_cri_hat[:, 4]
            rvds_dif = flux_cri_hat[:, 5]
            rnds_dir = flux_cri_hat[:, 6]
            rnds_dif = flux_cri_hat[:, 7]

            sw_loss_dirdif = torch.mean(torch.abs(rvds_dir + rvds_dif + rnds_dir + rnds_dif - rsds))
            self.log(f'{v}sw_loss_dirdif', sw_loss_dirdif, sync_dist=True)
        elif "LW" in self.model_type:
            rlds = flux_cri_hat[:,0] + dscs # all-sky flux
            rlut = flux_cri_hat[:,1] + utcs # all-sky flux
            ts_rad = x[:,n*l]
            sig = 5.670374419e-8
            em = 0.996
            rlus = em*sig*ts_rad**4
            fnet_flux = (-rlut-(rlds-rlus)) + 1.41 # denormalized net-flux
            # constant -1.41 is the bias that is also in the dataset
            # due to correction by surface scheme 
        energy =  torch.mean(torch.abs(fnet_flux - fnet_hr))
        self.log(f'{v}energy_loss', energy, sync_dist=True, prog_bar=True)
        # -------------

        # weighting energy loss term
        steps_per_epoch = self.trainer.num_training_batches
        start_step = 100 * steps_per_epoch
        n_steps = 10 * steps_per_epoch
        step = self.trainer.global_step
        weight = 0
        if step > start_step:
            weight = np.minimum( 1.e-8*10**((step-start_step)/n_steps), 1.e-1)
            loss = mse + mae + energy * weight
            # increase the weight of the energy loss as the model converges
            # a factor of 10 every 100 epochs
        else:
           loss = mse + mae 
        self.log(f'{v}energy_weight', weight, sync_dist=True)
        return loss


    def on_after_backward(self):
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:  # log every n steps
            for name, params in self.named_parameters():
                if params.grad is not None and torch.any(params.grad):
                    self.logger.experiment.add_histogram(name, params.grad, self.trainer.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,  patience=20, threshold=0.0001, threshold_mode='rel')

        return {"optimizer":optimizer, "lr_scheduler":{"scheduler": scheduler, "monitor": "val_mse_loss"}}
