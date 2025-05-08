import os

import argparse
import logging
import platform
from pathlib import Path

import torch
from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")

from collections import OrderedDict
from AR.utils import get_newest_ckpt
from process_ckpt import my_save

class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Text2SemanticLightningModule):
        if self._should_save_on_train_epoch_end(trainer):
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, self._monitor_candidates(trainer))

                if self.if_save_latest == True:
                    current_ckpt_files = sorted([f for f in os.listdir(self.dirpath) if f.endswith(".ckpt")],
                                                     key=lambda x: os.path.getmtime(os.path.join(self.dirpath, x)))
                    if len(current_ckpt_files) > 1:
                            for old_ckpt in current_ckpt_files[:-1]:
                                try:
                                    os.remove(os.path.join(self.dirpath, old_ckpt))
                                except Exception as e:
                                    pass

                if self.if_save_every_weights == True:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    dictt = trainer.lightning_module.state_dict()
                    for key in dictt:
                        if key in dictt:
                           try:
                               to_save_od["weight"][key] = dictt[key].half().cpu()
                           except Exception as e:
                                pass

                    to_save_od["config"] = self.config
                    to_save_od["info"] = "GPT-e%s" % (trainer.current_epoch + 1)

                    try:
                        Path(self.half_weights_save_dir).mkdir(parents=True, exist_ok=True)
                        my_save(
                            to_save_od,
                            "%s/%s-e%s.ckpt"
                            % (
                                self.half_weights_save_dir,
                                self.exp_name,
                                trainer.current_epoch + 1,
                            ),
                        )
                    except Exception as e:
                        pass


def main(args):
    config = load_yaml_config(args.config_file)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["train"]["seed"], workers=True)

    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )

    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)

    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["USE_LIBUV"] = "0"

    try:
        if ipex is not None and hasattr(torch, 'xpu'):
             is_xpu_avail_now = torch.xpu.is_available()
             xpu_dev_count = 0
             if is_xpu_avail_now:
                 try:
                    xpu_dev_count = torch.xpu.device_count()
                 except Exception as e:
                     pass
        else:
            pass
    except Exception as e:
        pass

    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        limit_val_batches=0,
        benchmark=False,
        fast_dev_run=False,
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        use_distributed_sampler=False,
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(config, output_dir)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
    )

    ckpt_path = None
    try:
        if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
             newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
             if newest_ckpt_name:
                 ckpt_path = os.path.join(ckpt_dir, newest_ckpt_name)
             else:
                 pass
        else:
             pass

    except Exception as e:
        ckpt_path = None

    trainer.fit(model, data_module, ckpt_path=ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="configs/s1longer.yaml",
        help="path of config file",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Arguments: {args}")
    main(args)