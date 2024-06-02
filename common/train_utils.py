
from pathlib import Path
import os
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from accelerate.utils import ProjectConfiguration, set_seed
import diffusers
from diffusers.utils.torch_utils import is_compiled_module
import wandb
import logging
import math
import sys
sys.path.append('..')
import random
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL
import copy
from tqdm import tqdm
from common.models import ViT
import torchvision
import numpy as np
import faiss


def get_kmeans_clusters(latents=None, dataset_name="cifar", k=256, image_size=32, num_images=2000, verbose=True, niter=100):
    transform = T.Compose(
        [T.ToTensor(),
         T.Resize(image_size),
         T.CenterCrop(image_size),]
    )

    dataset_classes = {
        "cifar": torchvision.datasets.CIFAR10,
        "celeb-a": torchvision.datasets.CelebA,
        "flowers102": torchvision.datasets.Flowers102,
    }
    dataset_cls = dataset_classes[dataset_name]
    dataset = dataset_cls(root='./data', #split='train',
                                 download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=64,
        num_workers=4,
         pin_memory=False
    )

    images = []
    for i, x in enumerate(dataloader):
        images.extend(x[0].chunk(x[0].size(0)))
        if len(images) >= num_images:
            break
    images = torch.cat(images, 0).permute(0, 2, 3, 1).reshape(-1, 3).numpy()
    c = 3 if dataset_name is not None else latents.shape[-1]
        
    kmeans = faiss.Kmeans(c, k, niter=niter, verbose=verbose)
    kmeans.train(images)

    return kmeans.centroids


def save_model(model, accelerator, save_path, args, logger, keyword="lora"):
    state_dict = {k:v for k,v in unwrap_model(accelerator, model).state_dict().items() if keyword in k}

    full_state_dict ={
        "state_dict": state_dict,
    }

    torch.save(full_state_dict, save_path)
    logger.info(f"Saved state to {save_path}")

def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def log_validation(
    model,
    args,
    accelerator,
    epoch,
    logger,
):
    logger.info(f"Running validation... \n Generating {args.num_validation_images} images")

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    images = []
    with torch.cuda.amp.autocast():
        images = model.sample(batch_size=args.num_validation_images, num_steps=args.num_validation_steps)#generator=generator)
        images.extend(images)

    for tracker in accelerator.trackers:
        if args.use_wandb:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}") for i, image in enumerate(images)
                        ]
                    }
                )

    torch.cuda.empty_cache()

    return images


def init_train_basics(args, logger):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    args.weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        args.weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        args.weight_dtype = torch.bfloat16

    # Enable TF32 for faster training on Ampere GPUs,
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    return accelerator


def load_models(args, accelerator):
    # Load the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="tokenizer",
    #     revision=args.revision,
    #     use_fast=False,
    # )

    # Load scheduler and models
    # text_encoder = transformers.CLIPTextModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    # ).requires_grad_(False).to(accelerator.device, dtype=weight_dtype)
    # vae = AutoencoderKL.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    # ).requires_grad_(False).to(accelerator.device, dtype=weight_dtype)

    most_common_classes=None
    if not args.per_channel:
        most_common_classes = get_kmeans_clusters(dataset_name=args.dataset_name, k=args.quant_k, image_size=args.resolution, num_images=2000, verbose=True, niter=100)

    model = ViT(
        image_size=args.resolution,
        num_classes=args.quant_k,
        dim=args.dim,
        time_dim=args.time_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        per_channel=args.per_channel,
        most_common_classes=most_common_classes
    ).to(accelerator.device)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    return model


def get_optimizer(args, params_to_optimize, accelerator):
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    return optimizer, lr_scheduler


def get_dataset(args):
    transform = T.Compose(
        [T.ToTensor(),
         T.Resize(args.resolution),
         T.CenterCrop(args.resolution),]
    )

    dataset_classes = {
        "cifar": torchvision.datasets.CIFAR10,
        "celeb-a": torchvision.datasets.CelebA,
        "flowers102": torchvision.datasets.Flowers102,
    }
    dataset_cls = dataset_classes[args.dataset_name]
    train_dataset = dataset_cls(root='./data', #split='train',
                                 download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
         pin_memory=args.pin_memory
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    return train_dataset, train_dataloader, num_update_steps_per_epoch


default_arguments = dict(
    text_encoder_path="runwayml/stable-diffusion-v1-5",
    dataset_name = "cifar", # "cifar" or "celeb-a" or "flowers102"
    num_validation_images=4,
    output_dir="model-output",
    seed=None,
    resolution=32,
    train_batch_size=16,
    max_train_steps=1250,
    validation_steps=250,
    checkpointing_steps=500,
    resume_from_checkpoint=None,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=6.0e-5,
    lr_scheduler="linear",
    lr_warmup_steps=50,
    lr_num_cycles=1,
    lr_power=1.0,
    dataloader_num_workers=4,
    use_8bit_adam=False,
    adam_beta1=0.9,
    adam_beta2=0.99,
    adam_weight_decay=1e-2,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    report_to="wandb",
    mixed_precision="bf16",
    allow_tf32=True,
    logging_dir="logs",
    local_rank=-1,
    num_processes=1,
    use_wandb=True,
    pin_memory=True,


    image_size=32,
    quant_k=4096,
    dim=512,
    time_dim=None,
    depth=12,
    num_heads=1,
    mlp_ratio=2.0,
    per_channel=False,
    num_validation_steps=50,

    mask_p=0.75,
    mutate_p=0.15,
)


def resume_model(model, path, accelerator):
    accelerator.print(f"Resuming from checkpoint {path}")
    global_step = int(path.split("-")[-1])
    state_dict = torch.load(path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    return global_step


def more_init(accelerator, args, train_dataloader, train_dataset, logger, num_update_steps_per_epoch, global_step, wandb_name="a"):
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        if args.use_wandb:
            accelerator.init_trackers(wandb_name, config=tracker_config)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    initial_global_step = global_step
    first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    return global_step, first_epoch, progress_bar