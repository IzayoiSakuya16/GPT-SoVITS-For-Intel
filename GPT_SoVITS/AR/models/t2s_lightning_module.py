import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
from typing import Dict

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer

from AR.models.t2s_model import Text2SemanticDecoder
from AR.modules.lr_schedulers import WarmupCosineLRSchedule
from AR.modules.optim import ScaledAdam

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None


class Text2SemanticLightningModule(LightningModule):
    def __init__(self, config, output_dir, is_train=True):
        super().__init__()
        self.config = config
        self.top_k = 3
        self.model = Text2SemanticDecoder(config=config, top_k=self.top_k)

        pretrained_s1 = config.get("pretrained_s1")
        if pretrained_s1 and is_train:
            try:
                load_info = self.load_state_dict(
                    torch.load(
                        pretrained_s1,
                        map_location="cpu",
                    )["weight"],
                    strict=False
                )
            except Exception as e:
                pass

        if is_train:
            self.automatic_optimization = False
            self.save_hyperparameters()
            self.eval_dir = output_dir / "eval"
            self.eval_dir.mkdir(parents=True, exist_ok=True)


    def on_train_start(self):
        target_device = 'cpu'

        if hasattr(torch, 'xpu'):
            try:
                is_xpu_avail = torch.xpu.is_available()
                if is_xpu_avail:
                    try:
                        xpu_count = torch.xpu.device_count()
                        if xpu_count > 0:
                            target_device = 'xpu:0'
                            self.to(target_device)
                        else:
                             target_device = 'cpu'
                    except Exception as e:
                        target_device = 'cpu'
                else:
                    target_device = 'cpu'
            except Exception as e:
                 target_device = 'cpu'
        else:
             target_device = 'cpu'


    def training_step(self, batch: Dict, batch_idx: int):
        try:
            if isinstance(batch, dict):
                 batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) and v.device != self.device else v for k, v in batch.items()}

        except Exception as e:
             pass

        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        forward = self.model.forward if self.config["train"].get("if_dpo", False) == True else self.model.forward_old
        loss, acc = forward(
            batch["phoneme_ids"],
            batch["phoneme_ids_len"],
            batch["semantic_ids"],
            batch["semantic_ids_len"],
            batch["bert_feature"],
        )

        self.manual_backward(loss)

        if batch_idx > 0 and batch_idx % 4 == 0:
            opt.step()
            opt.zero_grad()
            scheduler.step()

        self.log(
            "total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "lr",
            scheduler.get_last_lr()[0],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"top_{self.top_k}_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def validation_step(self, batch: Dict, batch_idx: int):
        return

    def configure_optimizers(self):
        model_parameters = self.model.parameters()
        parameters_names = []
        parameters_names.append([name_param_pair[0] for name_param_pair in self.model.named_parameters()])

        lm_opt = ScaledAdam(
            model_parameters,
            lr=0.01,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )

        scheduler = WarmupCosineLRSchedule(
            lm_opt,
            init_lr=self.config["optimizer"]["lr_init"],
            peak_lr=self.config["optimizer"]["lr"],
            end_lr=self.config["optimizer"]["lr_end"],
            warmup_steps=self.config["optimizer"]["warmup_steps"],
            total_steps=self.config["optimizer"]["decay_steps"],
        )

        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }

    def setup(self, stage: str):
        super().setup(stage)

        if stage == 'fit' and self.device.type == 'xpu':
            try:
                if ipex is not None:
                    pass
                else:
                     pass
            except Exception as e:
                pass