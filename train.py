import torch
torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)
import random
random.seed(123456)
import numpy as np
np.random.seed(123456)
import lightning as pl
from model import LDRNet
from data import DocDataModule
import configs
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
# from callbacks import MyPrintingCallback, EarlyStopping
# from pytorch_lightning.profilers import PyTorchProfiler

torch.set_float32_matmul_precision("medium") # to make lightning happy


if __name__ == "__main__":
    logger = TensorBoardLogger("logs", name = "test_v0")
    checkpoint_callback = ModelCheckpoint(dirpath="haha", save_top_k=3, monitor="val_loss")
    
    trainer = pl.Trainer(
        logger = logger,
        accelerator="gpu", 
        devices=1, 
        min_epochs=1, 
        max_epochs=1000, 
        precision='16-mixed',
        default_root_dir="test",
        check_val_every_n_epoch=5,
        enable_checkpointing = True,
        callbacks = [checkpoint_callback]
    )

    model = LDRNet(configs.n_points, lr = configs.lr, dropout = 0.2)

    # model.load_from_checkpoint("checkpoints/lightning_logs/version_3/checkpoints/epoch=29-step=13650.ckpt")
    
    dm = DocDataModule(
        json_path="/notebooks/haha.json",
        data_dir="/notebooks/sample_dataset",
        batch_size=configs.batch_size,
        num_workers=configs.num_workers
    )
    
    # tuner = pl.pytorch.tuner.tuning.Tuner(trainer)
    # lr_finder = tuner.lr_find(model, dm)
    
    trainer.fit(model, dm)
    # trainer.validate(model, dm)
    # trainer.test(model, dm, ckpt_path="checkpoints/lightning_logs/version_3/checkpoints/epoch=29-step=13650.ckpt")

