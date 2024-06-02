#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm

import train_utils
from train_utils import (
    init_train_basics,
    log_validation,
    save_model,
    unwrap_model,
    default_arguments,
    load_models,
    get_optimizer,
    get_dataset,
    more_init,
    resume_model
)
from types import SimpleNamespace

def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    accelerator = init_train_basics(args, logger)

    model = load_models(args, accelerator)
    optimizer, lr_scheduler = get_optimizer(args, list(model.parameters()), accelerator)
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = resume_model(model, args.resume_from_checkpoint, accelerator)

    global_step, first_epoch, progress_bar = more_init(accelerator, args, train_dataloader, train_dataset, 
                                                    logger, num_update_steps_per_epoch, global_step, wandb_name="pixel_muse")

    grad_norm = None
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # model_input = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                orig_ids = model.tokenizer.encode(batch[0])
                ids = orig_ids.clone()
                mask_mask = torch.bernoulli(torch.full(ids.shape, args.mask_p)).bool()
                mask_token = torch.full_like(ids, model.tokenizer.mask_token_id)
                ids[mask_mask] = mask_token[mask_mask]

                mutate_mask = torch.bernoulli(torch.full(ids.shape, args.mutate_p)).bool()
                mutate_token = torch.randint_like(ids, 0, model.tokenizer.vocab_size)
                ids[mutate_mask] = mutate_token[mutate_mask]

                embs = model.tokenizer.to_embs(ids)
                preds = model(embs)
                loss = torch.nn.functional.cross_entropy(preds.view(-1, preds.size(-1)), orig_ids.reshape(-1))

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_model(model, accelerator,save_path, args, logger)
                        

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "grad_norm": grad_norm}
            progress_bar.set_postfix(**logs)
            if args.use_wandb:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if global_step % args.validation_steps == 0 and global_step > 0:
                    images = log_validation(
                        model,
                        args,   
                        accelerator,
                        epoch=epoch,
                        logger=logger,
                    )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_model(model, accelerator, save_path, args, logger)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)