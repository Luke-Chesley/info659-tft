from model import build_tft,build_time_series_ds
from config import get_config
import torch

def train_model(config):
    train_dataloader, val_dataloader, _,_ = build_time_series_ds(config)

    trainer, tft,_ = build_tft(config)
    torch.set_float32_matmul_precision('high')
    trainer.fit(
        tft,
        train_dataloader,
        val_dataloader,
        #ckpt_path="m5/checkpoints/6/epoch=7-val_loss=455.33-6.ckpt",
    )


if __name__ == "__main__":
    config = get_config()
    train_model(config)
