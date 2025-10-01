#import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning import callbacks as cbs
from pytorch_lightning.loggers import TensorBoardLogger

# Custom imports
import config
from utils.quick_helpers import load_from_checkpoint
from preprocessing.load_data import PreBatchedDataset

def main():
    torch.set_float32_matmul_precision('medium')
    
    # Load config and data
    print("loading config and data...")
    train_args = config.setup_args_and_load_data()
    print(train_args.folder)
    print("*"*50)
    for k in train_args._get_kwargs():
        if "norm" in k[0] or "grid" in k[0]:
            continue
        print("* ", k)    
    print("*"*50)
    
    model = config.create_model(train_args, extra_shape=train_args.extra_shape)

    if train_args.pretrained:
        print("loading pretrained model")
        model.load_state_dict(torch.load(train_args.pretrained_path))

    print(str(model))
    # callbacks
    early_stopping = cbs.EarlyStopping(
        monitor='val_mse_mae_loss', 
        min_delta=0.00001, 
        patience=10, 
        mode='min', 
        check_on_train_epoch_end=False
    )
    lr_monitor = cbs.LearningRateMonitor(logging_interval='epoch')
    checkpoint = cbs.ModelCheckpoint(
        dirpath=train_args.checkpoint_path, 
         monitor='val_mse_mae_loss',
         mode='min', 
         save_top_k=1, 
         save_last=True, 
         filename='model_{epoch}'
    )
    logger = TensorBoardLogger(
        "lightning_logs", 
        name = train_args.folder, 
        version = train_args.model_type
    )
    
    # trainer
    trainer = L.Trainer(
        max_epochs=train_args.train_epochs, 
        accelerator="cpu" if train_args.dev else "gpu", 
        devices=1, 
        num_sanity_val_steps=0, 
        log_every_n_steps=1682, 
        profiler="simple",
        callbacks=[early_stopping, lr_monitor, checkpoint],
        fast_dev_run=train_args.dev, 
        logger=logger
    )

    train_loader = DataLoader(
        PreBatchedDataset(train_args.coarse_train, batch_size = train_args.batch_size),
        batch_size=None,
        shuffle=False,
    )	
    
    val_loader = DataLoader(
        PreBatchedDataset(train_args.coarse_val,  batch_size = train_args.batch_size),
        batch_size=None,
    )
    
    if train_args.checkpoint:
        ckpt_path = train_args.checkpoint_path + "last.ckpt"
        trainer.fit(
            model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader, 
            ckpt_path=ckpt_path
        )
    else: 
        trainer.fit(
            model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader
        )
    print(model.log)

    # Save model
    torch.save(model.state_dict(), train_args.model_path)
    
    # Save model without extra shape for inference
    model = config.load_model(train_args, 0)
    # create a trainer if it does not exist, so lightning can save the model
    model.trainer = L.Trainer()
    print(str(model))

    # saving jit model for online coupling
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
        scripted_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
        scripted_model.save(f"{train_args.save_folder}{train_args.model_type}_scripted_model_{device}.pt")
    device = "cpu"
    model.to(device)
    scripted_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
    scripted_model.save(f"{train_args.save_folder}{train_args.model_type}_scripted_model_{device}.pt")

if __name__ == '__main__':
    main()

