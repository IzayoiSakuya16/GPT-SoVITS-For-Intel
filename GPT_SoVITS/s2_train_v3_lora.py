import warnings

warnings.filterwarnings("ignore")
import os

import utils

hps = utils.get_hparams(stage=2)
#os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")
import logging

import torch

from torch.cuda.amp import GradScaler, autocast
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
from collections import OrderedDict as od
# from random import randint

from module import commons
from module.data_utils import (
    # DistributedBucketSampler,
    TextAudioSpeakerCollateV3,
    TextAudioSpeakerLoaderV3,
    TextAudioSpeakerCollateV4,
    TextAudioSpeakerLoaderV4,
)
from module.models import (
    SynthesizerTrnV3 as SynthesizerTrn,
)
from peft import LoraConfig, get_peft_model
from process_ckpt import savee


# 设置浮点精度，对于XPU可能也有效果
torch.set_float32_matmul_precision("medium")
global_step = 0

# device = "cpu"  # cuda以外的设备，等mps优化后加入


def main():
    # 检测可用设备
    if torch.xpu.is_available():
        device = torch.xpu.current_device() # 使用当前的 XPU 设备
        print(f"Using Intel XPU device: {device}")
        torch.xpu.set_device(device) # 显式设置设备
        n_gpus = 1 # 单卡模式
    elif torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Using CUDA device: {device}")
        torch.cuda.set_device(device)
        n_gpus = 1
    else:
        device = "cpu"
        print("No accelerator device available, using CPU.")
        n_gpus = 1
    run(device, hps, n_gpus)


def run(device, hps, n_gpus):
    global global_step, no_grad_names, save_root, lora_rank
    logger = utils.get_logger(hps.data.exp_dir)
    logger.info(hps)
    writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))
    TextAudioSpeakerLoader = TextAudioSpeakerLoaderV3 if hps.model.version == "v3" else TextAudioSpeakerLoaderV4
    TextAudioSpeakerCollate = TextAudioSpeakerCollateV3 if hps.model.version == "v3" else TextAudioSpeakerCollateV4
    train_dataset = TextAudioSpeakerLoader(hps.data)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn,
        # batch_sampler=train_sampler,
        batch_size=hps.train.batch_size,
        shuffle=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    save_root = "%s/logs_s2_%s_lora_%s" % (hps.data.exp_dir, hps.model.version, hps.train.lora_rank)
    os.makedirs(save_root, exist_ok=True)
    lora_rank = int(hps.train.lora_rank)
    lora_config = LoraConfig(
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights=True,
    )

    def get_model(hps):
        return SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )

    def get_optim(net_g):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net_g.parameters()),  ###默认所有层lr一致
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

     # 修改 model2cuda 函数以支持 XPU 或 CUDA
    def model2device(net_g, device):
        net_g = net_g.to(device)
        return net_g

    try:  # 如果能加载自动resume
        net_g = get_model(hps)
        net_g.cfm = get_peft_model(net_g.cfm, lora_config)
        net_g = model2device(net_g, device)
        optim_g = get_optim(net_g)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(save_root, "G_*.pth"),
            net_g,
            optim_g,
        )
        epoch_str += 1
        if global_step == 0 and epoch_str > 1:  # 如果 epoch_str > 1，说明成功加载了，重新计算 global_step
            global_step = (epoch_str - 1) * len(train_loader)
        elif global_step == 0:  # 如果是首次运行或加载失败从头开始
            global_step = 0

    except Exception as e:  # 捕获更广泛的异常，并打印出来帮助调试
        import traceback
        traceback.print_exc()  # 打印详细错误信息
        logger.info("Could not load latest checkpoint, starting from epoch 1 or pretrained if available.")
        logger.error(f"Loading error: {e}")

        epoch_str = 1
        global_step = 0
        net_g = get_model(hps)
        if (
                hps.train.pretrained_s2G != ""
                and hps.train.pretrained_s2G != None
                and os.path.exists(hps.train.pretrained_s2G)
        ):
            logger.info("loaded pretrained %s" % hps.train.pretrained_s2G)
            # load_state_dict 默认map_location='cpu'，这通常是加载权重的安全方式，然后再将其发送到设备
            try:
                net_g.load_state_dict(
                    torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                    strict=False,
                )
            except Exception as e:
                logger.error(f"Error loading pretrained model: {e}")
                traceback.print_exc()
                # 根据错误类型决定是否退出或继续 (例如文件损坏或格式不匹配)
                # exit(1) # 如果预训练模型加载是必须的，可以在这里退出

        net_g.cfm = get_peft_model(net_g.cfm, lora_config)
        net_g = model2device(net_g, device)
        optim_g = get_optim(net_g)

        # ... (rest of the code seems mostly fine regarding device handling) ...

    no_grad_names = set()
    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            no_grad_names.add(name)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    for _ in range(epoch_str - 1):
        scheduler_g.step()

    # GradScaler 对于 XPU 也可用，前提是 IPEX 支持混合精度
    scaler = GradScaler(enabled=hps.train.fp16_run)

    net_d = optim_d = scheduler_d = None  # Discriminator seems unused in this script?

    for epoch in range(epoch_str, hps.train.epochs + 1):

        train_and_evaluate(
            epoch,
            hps,
            net_g,  # 直接传递模型
            optim_g,  # 直接传递优化器
            scheduler_g,  # 直接传递调度器
            scaler,
            train_loader,
            logger,
            writer,  # <--- 移除 writer_eval
            device  # <--- 传递设备参数
        )
        scheduler_g.step()
        print(f"Epoch {epoch} finished.")

    print("training done")


def train_and_evaluate(epoch, hps, net_g, optim_g, scheduler_g, scaler, train_loader, logger, writer, device):
    global global_step

    net_g.train()
    data_iterator = tqdm(train_loader, desc=f"Epoch {epoch}")
    current_device = device

    for batch_idx, (ssl, spec, mel, ssl_lengths, spec_lengths, text, text_lengths, mel_lengths) in enumerate(
            data_iterator
    ):
        # 将数据移动到当前进程的设备上
        spec, spec_lengths = (
            spec.to(current_device, non_blocking=True),
            spec_lengths.to(current_device, non_blocking=True),
        )
        mel, mel_lengths = mel.to(current_device, non_blocking=True), mel_lengths.to(current_device, non_blocking=True)
        ssl = ssl.to(current_device, non_blocking=True)
        ssl.requires_grad = False
        text, text_lengths = (
            text.to(current_device, non_blocking=True),
            text_lengths.to(current_device, non_blocking=True),
        )

        # autocast 对于 XPU 也可用，前提是 IPEX 支持
        with autocast(enabled=hps.train.fp16_run and (torch.xpu.is_available() or torch.cuda.is_available())):
            cfm_loss = net_g(
                ssl,
                spec,
                mel,
                ssl_lengths,
                spec_lengths,
                text,
                text_lengths,
                mel_lengths,
                use_grad_ckpt=hps.train.grad_ckpt,
            )
            loss_gen_all = cfm_loss

        # Optimizer step (standard PyTorch, should work with IPEX)
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(),
                                               None)  # Ensure grad_norm_g calculation works with XPU tensors
        scaler.step(optim_g)
        scaler.update()

        if global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            losses = [cfm_loss]

            log_message = f"Epoch: {epoch} [{100.0 * batch_idx / len(train_loader):.0f}%] | Global Step: {global_step} | LR: {lr:.6f} | Loss: {loss_gen_all.item():.4f} | Grad Norm G: {grad_norm_g:.4f}"
            logger.info(log_message)

            scalar_dict = {"loss/g/total": loss_gen_all.item(), "learning_rate": lr, "grad_norm_g": grad_norm_g}
            utils.summarize(
                writer=writer,
                global_step=global_step,
                scalars=scalar_dict,
            )

        global_step += 1
        # 在 epoch 结束时，所有进程同步后再保存模型
    if epoch % hps.train.save_every_epoch == 0:
        if hps.train.if_save_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(save_root, "G_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(save_root, "G_{}.pth".format(233333333333)),  # Using magic number for latest
            )
        if hps.train.if_save_every_weights == True:
            # Get model state_dict (没有 DDP wrapper 了)
            ckpt = net_g.state_dict()

            sim_ckpt = od()
            for key in ckpt:
                # Check if key needs to be saved (not in no_grad_names)
                if key not in no_grad_names:
                    # Ensure tensors are on CPU and potentially converted to half precision before saving
                    sim_ckpt[key] = ckpt[key].half().cpu()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        sim_ckpt,
                        hps.name + "_e%s_s%s_l%s" % (epoch, global_step, lora_rank),
                        epoch,
                        global_step,
                        hps, cfm_version=hps.model.version,
                        lora_rank=lora_rank,
                    ),
                )
            )

    logger.info("====> Epoch: {}".format(epoch))

if __name__ == "__main__":
    main()
