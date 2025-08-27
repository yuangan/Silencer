# @ Haotian Xue 2023
# accelerated version: mist v3
# feature 1: SDS
# feature 2: Diff-PGD

import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
import PIL
from PIL import Image
import ssl
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ldm.util import instantiate_from_config
from advertorch.attacks import LinfPGDAttack
from attacks import Linf_PGD, SDEdit, Linf_PGD_Hallo
import time
import glob
import hydra
from utils import mp, si, cprint

import argparse
import copy
import logging
import math
import os
import random
import warnings
from datetime import datetime
from typing import List, Tuple

import diffusers
import mlflow
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.image_processor import VaeImageProcessor
from einops import rearrange, repeat

from hallo.animate.face_animate import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.datasets.talk_video import TalkingVideoDataset
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.mutual_self_attention import ReferenceAttentionControl, torch_dfs
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.util import (compute_snr, delete_additional_ckpt,
                              import_filename, init_output_dir,
                              load_checkpoint, save_checkpoint,
                              seed_everything, tensor_to_video)



ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_config(config_path: str) -> dict:
    """
    Loads the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """

    if config_path.endswith(".yaml"):
        return OmegaConf.load(config_path)
    if config_path.endswith(".py"):
        return import_filename(config_path).cfg
    raise ValueError("Unsupported format for config file")

def process_audio_emb(audio_emb):
    """
    Process the audio embedding to concatenate with other tensors.

    Parameters:
        audio_emb (torch.Tensor): The audio embedding tensor to process.

    Returns:
        concatenated_tensors (List[torch.Tensor]): The concatenated tensor list.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb

class HalloRefNet(nn.Module):
    def __init__(self, cfg):
        """
        Initializes the trainer with the given configuration.

        Args:
            cfg (argparse.Namespace): The configuration containing parameters for training.
        """
        super().__init__()
        self.cfg = load_config(cfg)
        
        # Initialize models, optimizers, and schedulers
        self.initialize_models()

        # Load data
        self.train_dataloader = self.prepare_data()

    def initialize_models(self):
        """
        Initializes the models and loads the pretrained weights from stage 1.
        """
        # Weight Dtype
        weight_dtype = self.get_weight_dtype(self.cfg.weight_dtype)

        # Load Models
        self.vae = AutoencoderKL.from_pretrained(self.cfg.vae_model_path).to(
            "cuda", dtype=weight_dtype)
        self.reference_unet = UNet2DConditionModel.from_pretrained(
            self.cfg.base_model_path, subfolder="unet").to(device="cuda", dtype=weight_dtype)
        self.imageproj = ImageProjModel(
            cross_attention_dim=768, # self.denoising_unet.config.cross_attention_dim
            clip_embeddings_dim=512, clip_extra_context_tokens=4).to(device="cuda", dtype=weight_dtype)

        self.vae_scale_factor: int = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True,
        )

    def prepare_data(self):
        """
        Prepares the data loader for training.

        Returns:
            DataLoader: The training data loader.
        """
        train_dataset = TalkingVideoDataset(
            img_size=(self.cfg.data.train_width, self.cfg.data.train_height),
            sample_rate=self.cfg.data.sample_rate,
            n_sample_frames=self.cfg.data.n_sample_frames,
            n_motion_frames=self.cfg.data.n_motion_frames,
            audio_margin=self.cfg.data.audio_margin,
            data_meta_paths=self.cfg.data.train_meta_paths,
            wav2vec_cfg=self.cfg.wav2vec_config,
        )
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg.data.train_bs, shuffle=True, num_workers=16)

    def get_weight_dtype(self, weight_dtype):
        """
        Determines the appropriate weight data type based on the configuration.

        Args:
            weight_dtype (str): The dtype configuration ("fp16", "bf16", "fp32").

        Returns:
            torch.dtype: The corresponding torch dtype.
        """
        if weight_dtype == "fp16":
            return torch.float16
        elif weight_dtype == "bf16":
            return torch.bfloat16
        elif weight_dtype == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unsupported weight dtype: {weight_dtype}")

    def get_reference_output(
        self,
        ref_image: torch.Tensor,
        face_emb: torch.Tensor,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 3.5,
        return_latent = False,
    ) -> torch.Tensor:
        """
        获取 reference_unet 的输出。

        Args:
            ref_image (torch.Tensor): 参考图像，形状为 (batch_size, frames, channels, height, width)。
            face_emb (torch.Tensor): 面部嵌入特征，形状根据模型定义。
            width (int): 图像的宽度。
            height (int): 图像的高度。
            guidance_scale (float): 引导比例，通常大于 1.0。

        Returns:
            torch.Tensor: reference_unet 的输出。
        """
        # print(ref_image.shape, ref_image.max(), ref_image.min())
        ref_image = ref_image.unsqueeze(0).unsqueeze(0)
        device = ref_image.device
        batch_size = ref_image.shape[0]
        do_classifier_free_guidance = guidance_scale > 1.0

        # 准备编码器隐藏状态
        clip_image_embeds = torch.tensor(face_emb.reshape(1, -1)).to(self.imageproj.device, self.imageproj.dtype)
        encoder_hidden_states = self.imageproj(clip_image_embeds)
        if do_classifier_free_guidance:
            uncond_encoder_hidden_states = self.imageproj(torch.zeros_like(clip_image_embeds))
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)

        # 计算 ref_image_latents
        ref_image_tensor = rearrange(ref_image, "b f c h w -> (b f) c h w")
        ref_image_tensor = self.ref_image_processor.preprocess(ref_image_tensor, height=height, width=width)
        ref_image_tensor = ref_image_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # 这是一个常数，用于缩放
        
        if return_latent:
            return ref_image_latents

        # 准备时间步长 t
        t = torch.zeros(1, device=device, dtype=ref_image_latents.dtype)

        # 准备 reference_unet 的输入
        ref_image_latents_input = ref_image_latents.repeat(
            (2 if do_classifier_free_guidance else 1), 1, 1, 1
        )

        # 调用 reference_unet 并获取输出，((2, 320, 64, 64), )1. 2.
        reference_output = self.reference_unet(
            ref_image_latents_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )

        return reference_output


    def forward(self, batch):
        """
        Forward pass through the model.

        Args:
            batch (dict): A batch of training data.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        # pixel_values_vid = batch["pixel_values_vid"].to(dtype=self.cfg.weight_dtype)

        # # Preprocess masks
        # pixel_values_face_mask = get_attention_mask(batch["pixel_values_face_mask"], self.cfg.weight_dtype)
        # pixel_values_lip_mask = get_attention_mask(batch["pixel_values_lip_mask"], self.cfg.weight_dtype)
        # pixel_values_full_mask = get_attention_mask(batch["pixel_values_full_mask"], self.cfg.weight_dtype)

        # with torch.no_grad():
        #     video_length = pixel_values_vid.shape[1]
        #     pixel_values_vid = rearrange(pixel_values_vid, "b f c h w -> (b f) c h w")
        #     latents = self.vae.encode(pixel_values_vid).latent_dist.sample()
        #     latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        #     latents = latents * 0.18215

        # # Add noise
        # noise = torch.randn_like(latents)
        # if self.cfg.noise_offset > 0:
        #     noise += self.cfg.noise_offset * torch.randn(
        #         (latents.shape[0], latents.shape[1], 1, 1, 1), device=latents.device)

        # bsz = latents.shape[0]
        # timesteps = torch.randint(0, self.train_noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

        # noisy_latents = self.train_noise_scheduler.add_noise(latents, noise, timesteps)

        # # Get model predictions and compute loss
        # model_pred = self.net(
        #     noisy_latents=noisy_latents,
        #     timesteps=timesteps,
        #     ref_image_latents=None,  # Replace with appropriate tensor
        #     face_emb=batch["face_emb"],
        #     mask=None,  # Replace with appropriate tensor
        #     full_mask=pixel_values_full_mask,
        #     face_mask=pixel_values_face_mask,
        #     lip_mask=pixel_values_lip_mask,
        #     audio_emb=batch["audio_tensor"].to(dtype=self.cfg.weight_dtype),
        #     uncond_img_fwd=random.random() < self.cfg.uncond_img_ratio,
        #     uncond_audio_fwd=random.random() < self.cfg.uncond_audio_ratio,
        # )
        # """
        # simple docstring to prevent pylint error
        # """
        # face_emb = self.imageproj(face_emb)
        # mask = mask.to(device="cuda")
        # mask_feature = self.face_locator(mask)
        # audio_emb = audio_emb.to(
        #     device=self.audioproj.device, dtype=self.audioproj.dtype)
        # audio_emb = self.audioproj(audio_emb)

        # # condition forward
        # if not uncond_img_fwd:
        #     ref_timesteps = torch.zeros_like(timesteps)
        #     ref_timesteps = repeat(
        #         ref_timesteps,
        #         "b -> (repeat b)",
        #         repeat=ref_image_latents.size(0) // ref_timesteps.size(0),
        #     )
        #     self.reference_unet(
        #         ref_image_latents,
        #         ref_timesteps,
        #         encoder_hidden_states=face_emb,
        #         return_dict=False,
        #     )
        #     self.reference_control_reader.update(self.reference_control_writer)

        # return loss

        print('Not Implemented')
        assert(0)

class Net(nn.Module):
    """
    The Net class defines a neural network model that combines a reference UNet2DConditionModel,
    a denoising UNet3DConditionModel, a face locator, and other components to animate a face in a static image.

    Args:
        reference_unet (UNet2DConditionModel): The reference UNet2DConditionModel used for face animation.
        denoising_unet (UNet3DConditionModel): The denoising UNet3DConditionModel used for face animation.
        face_locator (FaceLocator): The face locator model used for face animation.
        reference_control_writer: The reference control writer component.
        reference_control_reader: The reference control reader component.
        imageproj: The image projection model.
        audioproj: The audio projection model.

    Forward method:
        noisy_latents (torch.Tensor): The noisy latents tensor.
        timesteps (torch.Tensor): The timesteps tensor.
        ref_image_latents (torch.Tensor): The reference image latents tensor.
        face_emb (torch.Tensor): The face embeddings tensor.
        audio_emb (torch.Tensor): The audio embeddings tensor.
        mask (torch.Tensor): Hard face mask for face locator.
        full_mask (torch.Tensor): Pose Mask.
        face_mask (torch.Tensor): Face Mask
        lip_mask (torch.Tensor): Lip Mask
        uncond_img_fwd (bool): A flag indicating whether to perform reference image unconditional forward pass.
        uncond_audio_fwd (bool): A flag indicating whether to perform audio unconditional forward pass.

    Returns:
        torch.Tensor: The output tensor of the neural network model.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        reference_control_writer,
        reference_control_reader,
        imageproj,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        ref_image_latents: torch.Tensor,
        face_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        mask: torch.Tensor,
        full_mask: torch.Tensor,
        face_mask: torch.Tensor,
        lip_mask: torch.Tensor,
        uncond_img_fwd: bool = False,
        uncond_audio_fwd: bool = False,
    ):
        """
        simple docstring to prevent pylint error
        """
        face_emb = self.imageproj(face_emb)
        mask = mask.to(device="cuda")
        mask_feature = self.face_locator(mask)
        audio_emb = audio_emb.to(
            device=self.audioproj.device, dtype=self.audioproj.dtype)
        audio_emb = self.audioproj(audio_emb)

        # condition forward
        if not uncond_img_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            ref_timesteps = repeat(
                ref_timesteps,
                "b -> (repeat b)",
                repeat=ref_image_latents.size(0) // ref_timesteps.size(0),
            )
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=face_emb,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        if uncond_audio_fwd:
            audio_emb = torch.zeros_like(audio_emb).to(
                device=audio_emb.device, dtype=audio_emb.dtype
            )

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            mask_cond_fea=mask_feature,
            encoder_hidden_states=face_emb,
            audio_embedding=audio_emb,
            full_mask=full_mask,
            face_mask=face_mask,
            lip_mask=lip_mask
        ).sample

        return model_pred

def get_noise_scheduler(cfg: argparse.Namespace) -> Tuple[DDIMScheduler, DDIMScheduler]:
    """
    Create noise scheduler for training.

    Args:
        cfg (argparse.Namespace): Configuration object.

    Returns:
        Tuple[DDIMScheduler, DDIMScheduler]: Train noise scheduler and validation noise scheduler.
    """

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    return train_noise_scheduler, val_noise_scheduler

def get_attention_mask(mask: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    """
    Rearrange the mask tensors to the required format.

    Args:
        mask (torch.Tensor): The input mask tensor.
        weight_dtype (torch.dtype): The data type for the mask tensor.

    Returns:
        torch.Tensor: The rearranged mask tensor.
    """
    if isinstance(mask, List):
        _mask = []
        for m in mask:
            _mask.append(
                rearrange(m, "b f 1 h w -> (b f) (h w)").to(weight_dtype))
        return _mask
    mask = rearrange(mask, "b f 1 h w -> (b f) (h w)").to(weight_dtype)
    return mask

class HalloNet(nn.Module):
    def __init__(self, cfg, min_timesteps):
        """
        Initializes the trainer with the given configuration.

        Args:
            cfg (argparse.Namespace): The configuration containing parameters for training.
        """
        super().__init__()
        self.cfg = load_config(cfg)
        
        self.weight_dtype = self.get_weight_dtype(self.cfg.weight_dtype)

        # Initialize models, optimizers, and schedulers
        self.vae, self.imageproj, self.denoising_unet, self.net = self.initialize_models()

        # get noise scheduler
        self.train_noise_scheduler, self.val_noise_scheduler = get_noise_scheduler(self.cfg)

        # self.accelerator = Accelerator()
        # self.net = self.accelerator.prepare(self.net)

        self.tensor_result = [] # save temp reuslts for motion
        self.audio_emb = torch.load('./hallo/th1kh/th1kh_tmp/audio_emb/0001.pt')
        # self.audio_emb = torch.load('/home/gy/code/talking-head/hallo/celebahq_512_dataset/supercool/audio_emb/0002.pt')
        self.audio_emb = process_audio_emb(self.audio_emb)
        self.clip_length = 1
        self.min_timesteps = min_timesteps
        self.max_timesteps = self.min_timesteps + 100

        # structure_denoising_unet = torch_dfs(self.denoising_unet)

        # self.hidden_states_list = []
        # def hook_fn(module, input, output):
        #     self.hidden_states_list.append(output)
        
        # handle = model.transformer_layer.register_forward_hook(hook_fn)

    def initialize_models(self):
        cfg = self.cfg
        weight_dtype = self.weight_dtype

        # Create Models
        vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
            "cuda", dtype=weight_dtype
        )
        reference_unet = UNet2DConditionModel.from_pretrained(
            cfg.base_model_path,
            subfolder="unet",
        ).to(device="cuda", dtype=weight_dtype)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.base_model_path,
            cfg.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                cfg.unet_additional_kwargs),
            use_landmark=False
        ).to(device="cuda", dtype=weight_dtype)

        imageproj = ImageProjModel(
            cross_attention_dim=denoising_unet.config.cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=4,
        ).to(device="cuda", dtype=weight_dtype)
        face_locator = FaceLocator(
            conditioning_embedding_channels=320,
        ).to(device="cuda", dtype=weight_dtype)
        audioproj = AudioProjModel(
            seq_len=5,
            blocks=12,
            channels=768,
            intermediate_dim=512,
            output_dim=768,
            context_tokens=32,
        ).to(device="cuda", dtype=weight_dtype)

        # Do not Freeze
        vae.requires_grad_(False)
        imageproj.requires_grad_(False)
        reference_unet.requires_grad_(False)
        denoising_unet.requires_grad_(False)
        face_locator.requires_grad_(False)
        audioproj.requires_grad_(False)

        reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )

        net = Net(
            reference_unet,
            denoising_unet,
            face_locator,
            reference_control_writer,
            reference_control_reader,
            imageproj,
            audioproj,
        ).to(dtype=weight_dtype)


        audio_ckpt_dir = cfg.audio_ckpt_dir
        m,u = net.load_state_dict(
            torch.load(
                os.path.join(audio_ckpt_dir, "net.pth"),
                map_location="cpu",
            ),
        )
        assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
        print("loaded weight from ", os.path.join(audio_ckpt_dir, "net.pth"))

        if cfg.solver.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                reference_unet.enable_xformers_memory_efficient_attention()
                denoising_unet.enable_xformers_memory_efficient_attention()

            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if cfg.solver.gradient_checkpointing:
            reference_unet.enable_gradient_checkpointing()
            denoising_unet.enable_gradient_checkpointing()

        return vae, imageproj, denoising_unet, net

    def get_weight_dtype(self, weight_dtype):
        """
        Determines the appropriate weight data type based on the configuration.

        Args:
            weight_dtype (str): The dtype configuration ("fp16", "bf16", "fp32").

        Returns:
            torch.dtype: The corresponding torch dtype.
        """
        if weight_dtype == "fp16":
            return torch.float16
        elif weight_dtype == "bf16":
            return torch.bfloat16
        elif weight_dtype == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unsupported weight dtype: {weight_dtype}")

    def get_reference_output(
        self,
        ref_image: torch.Tensor,
        face_emb: torch.Tensor,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 3.5,
        return_latent = False,
    ) -> torch.Tensor:
        """
        获取 reference_unet 的输出。

        Args:
            ref_image (torch.Tensor): 参考图像，形状为 (batch_size, frames, channels, height, width)。
            face_emb (torch.Tensor): 面部嵌入特征，形状根据模型定义。
            width (int): 图像的宽度。
            height (int): 图像的高度。
            guidance_scale (float): 引导比例，通常大于 1.0。

        Returns:
            torch.Tensor: reference_unet 的输出。
        """
        ref_image = ref_image.unsqueeze(0).unsqueeze(0)
        device = ref_image.device
        batch_size = ref_image.shape[0]
        do_classifier_free_guidance = guidance_scale > 1.0

        # 准备编码器隐藏状态
        clip_image_embeds = torch.tensor(face_emb.reshape(1, -1)).to(self.imageproj.device, self.imageproj.dtype)
        encoder_hidden_states = self.imageproj(clip_image_embeds)
        if do_classifier_free_guidance:
            uncond_encoder_hidden_states = self.imageproj(torch.zeros_like(clip_image_embeds))
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)

        # 计算 ref_image_latents
        ref_image_tensor = rearrange(ref_image, "b f c h w -> (b f) c h w")
        ref_image_tensor = self.ref_image_processor.preprocess(ref_image_tensor, height=height, width=width)
        ref_image_tensor = ref_image_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # 这是一个常数，用于缩放
        if return_latent:
            return ref_image_latents

        # 准备时间步长 t
        t = torch.zeros(1, device=device, dtype=ref_image_latents.dtype)

        # 准备 reference_unet 的输入
        ref_image_latents_input = ref_image_latents.repeat(
            (2 if do_classifier_free_guidance else 1), 1, 1, 1
        )

        # 调用 reference_unet 并获取输出，((2, 320, 64, 64), )1. 2.
        reference_output = self.reference_unet(
            ref_image_latents_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )

        return reference_output

    def forward(self, batch):
        """
        Forward pass through the model.

        Args:
            batch (dict): A batch of training data.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        weight_dtype = self.weight_dtype
        # Train!
        # Convert videos to latent space
        # print(batch["image"].shape, batch["image"].max(), batch["image"].min()) # [3, 512, 512] 1.1206 -1.1255
        pixel_values_vid = batch["image"].unsqueeze(0).unsqueeze(0).to(weight_dtype) # TODO: replace with inferenced results next!
        if pixel_values_vid.shape[1] == 1:
            pixel_values_vid = pixel_values_vid.repeat(1, self.clip_length, 1, 1, 1)

        pixel_values_face_mask = batch["face_mask"]

        pixel_values_lip_mask = batch["lip_mask"]

        pixel_values_full_mask = batch["full_mask"]

        pixel_values_full_mask = [
            (mask.repeat(self.clip_length, 1))
            for mask in pixel_values_full_mask
        ]
        pixel_values_face_mask = [
            (mask.repeat(self.clip_length, 1))
            for mask in pixel_values_face_mask
        ]
        pixel_values_lip_mask = [
            (mask.repeat(self.clip_length, 1))
            for mask in pixel_values_lip_mask
        ]

        pixel_values_face_mask_ = []
        for mask in pixel_values_face_mask:
            pixel_values_face_mask_.append(
                mask.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype))
        pixel_values_face_mask = pixel_values_face_mask_
        pixel_values_lip_mask_ = []
        for mask in pixel_values_lip_mask:
            pixel_values_lip_mask_.append(
                mask.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype))
        pixel_values_lip_mask = pixel_values_lip_mask_
        pixel_values_full_mask_ = []
        for mask in pixel_values_full_mask:
            pixel_values_full_mask_.append(
                mask.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype))
        pixel_values_full_mask = pixel_values_full_mask_

        # [-1, 1] (1, 1, 3, 512, 512)
        # print('pixel_values_vid:', pixel_values_vid.max(), pixel_values_vid.min(), pixel_values_vid.shape)

        with torch.no_grad():
            video_length = pixel_values_vid.shape[1]
            pixel_values_vid = rearrange(
                pixel_values_vid, "b f c h w -> (b f) c h w"
            )
            latents = self.vae.encode(pixel_values_vid).latent_dist.sample()
            latents = rearrange(
                latents, "(b f) c h w -> b c f h w", f=video_length
            )
            latents = latents * 0.18215
        # print(latents.shape) # [1, 4, 1, 64, 64]

        noise = torch.randn_like(latents)
        if self.cfg.noise_offset > 0:
            noise += self.cfg.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1, 1),
                device=latents.device,
            )

        bsz = latents.shape[0]
        # Sample a random timestep for each video
        timesteps = torch.randint(
            self.min_timesteps,
            self.max_timesteps,
            # self.train_noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # mask for face locator
        # print(batch["face_region"].shape) # [3, 512, 512]
        pixel_values_mask = (
            batch["face_region"].unsqueeze(0).unsqueeze(
                1).to(dtype=weight_dtype)
        )
        pixel_values_mask = repeat(
            pixel_values_mask,
            "b f c h w -> b (repeat f) c h w",
            repeat=video_length,
        )
        pixel_values_mask = pixel_values_mask.transpose(
            1, 2)

        uncond_img_fwd = False # False: use image condition
        uncond_audio_fwd = random.random() < self.cfg.uncond_audio_ratio

        # start_frame = random.random() < self.cfg.start_ratio
        
        ################# reference image process #####################
        pixel_values_ref_img = batch["image"].to(
            dtype=weight_dtype
        ).unsqueeze(0)
        
        # TODO: Modify here with inferenced results.
        if len(self.tensor_result) == 0:
            # The first iteration
            motion_zeros = pixel_values_ref_img.repeat(
                self.cfg.data.n_motion_frames, 1, 1, 1)
            motion_zeros = motion_zeros.to(
                dtype=pixel_values_ref_img.dtype, device=pixel_values_ref_img.device)
            pixel_values_ref_img = torch.cat(
                [pixel_values_ref_img, motion_zeros], dim=0)  # concat the ref image and the first motion frames
        else:
            motion_frames = self.tensor_result[-1][0]
            motion_frames = motion_frames.permute(1, 0, 2, 3)
            motion_frames = motion_frames[0-config.data.n_motion_frames:]
            motion_frames = motion_frames * 2.0 - 1.0
            motion_frames = motion_frames.to(
                dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat(
                [source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames

        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0) # [1 3 3 512 512]

        start_frame = True # attack start_frame
        # initialize the motion frames as zero maps
        if start_frame:
            pixel_values_ref_img[:, 1:] = 0.0

        ref_img_and_motion = rearrange(
            pixel_values_ref_img, "b f c h w -> (b f) c h w"
        )

        ref_image_latents = self.vae.encode(
            ref_img_and_motion
        ).latent_dist.sample()
        ref_image_latents = ref_image_latents * 0.18215
        image_prompt_embeds = batch["face_emb"].to(
            dtype=self.imageproj.dtype, device=self.imageproj.device
        ).reshape(1, -1)

        # add noise
        noisy_latents = self.train_noise_scheduler.add_noise(
            latents, noise, timesteps
        )

        # Get the target for loss depending on the prediction type
        if self.train_noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif self.train_noise_scheduler.prediction_type == "v_prediction":
            target = self.train_noise_scheduler.get_velocity(
                latents, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {self.train_noise_scheduler.prediction_type}"
            )
        start_idx = random.randint(self.cfg.data.n_motion_frames, self.audio_emb.shape[0]-self.clip_length)
        audio_tensor = self.audio_emb[start_idx:start_idx+self.clip_length].unsqueeze(0)

        # print('=============== check input shape ==============')
        # print('noisy_latents', noisy_latents.shape)
        # print('timesteps', timesteps.shape)
        # print('ref_image_latents', ref_image_latents.shape)
        # print('face_emb', image_prompt_embeds.shape)
        # print('mask', pixel_values_mask.shape)
        # print('full_mask', pixel_values_full_mask[0].shape)
        # print('face_mask', pixel_values_face_mask[0].shape)
        # print('lip_mask', pixel_values_lip_mask[0].shape)
        # print('audio_emb', audio_tensor.shape)
        # print(uncond_img_fwd)
        # print(uncond_audio_fwd)
        # assert(0)
        
        #### add xxx loss

        # ---- Forward!!! -----
        model_pred = self.net(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            ref_image_latents=ref_image_latents,
            face_emb=image_prompt_embeds,
            mask=pixel_values_mask,
            full_mask=pixel_values_full_mask,
            face_mask=pixel_values_face_mask,
            lip_mask=pixel_values_lip_mask,
            audio_emb=audio_tensor,
            uncond_img_fwd=uncond_img_fwd,
            uncond_audio_fwd=uncond_audio_fwd,
        )
        # print(model_pred.shape, model_pred.max(), model_pred.min())

        if self.cfg.snr_gamma == 0:
            loss = F.mse_loss(
                model_pred.float(),
                target.float(),
                reduction="mean",
            )
        else:
            snr = compute_snr(self.train_noise_scheduler, timesteps)
            if self.train_noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack(
                    [snr, self.cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )
            loss = F.mse_loss(
                model_pred.float(),
                target.float(),
                reduction="mean",
            )
            loss = (
                loss.mean(dim=list(range(1, len(loss.shape))))
                * mse_loss_weights
            ).mean()

        return loss

        # Gather the losses across all processes for logging (if we use distributed training).
        # avg_loss = self.accelerator.gather(
        #     loss.repeat(self.cfg.data.train_bs)).mean()
        # train_loss += avg_loss.item() / self.cfg.solver.gradient_accumulation_steps
        # return train_loss

        # Backpropagate
        # self.accelerator.backward(loss)


def load_image_from_path(image_path: str, input_size: int) -> PIL.Image.Image:
    """
    Load image form the path and reshape in the input size.
    :param image_path: Path of the input image
    :param input_size: The requested size in int.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    img = Image.open(image_path).resize((input_size, input_size),
                                        resample=PIL.Image.BICUBIC)
    return img


def load_model_from_config(config, ckpt, verbose: bool = False, device=0):
    """
    Load model from the config and the ckpt path.
    :param config: Path of the config of the SDM model.
    :param ckpt: Path of the weight of the SDM model
    :param verbose: Whether to show the unused parameters weight.
    :returns: A SDM model.
    """
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    # Support loading weight from NovelAI
    if "state_dict" in sd:
        import copy
        sd_copy = copy.deepcopy(sd)
        for key in sd.keys():
            if key.startswith('cond_stage_model.transformer') and not key.startswith('cond_stage_model.transformer.text_model'):
                newkey = key.replace('cond_stage_model.transformer', 'cond_stage_model.transformer.text_model', 1)
                sd_copy[newkey] = sd[key]
                del sd_copy[key]
        sd = sd_copy

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.cond_stage_model.to(device)
    model.eval()
    return model


class identity_loss(nn.Module):
    """
    An identity loss used for input fn for advertorch. To support semantic loss,
    the computation of the loss is implemented in class targe_model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x


class target_model(nn.Module):
    """
    A virtual model which computes the semantic and textural loss in forward function.
    """

    def __init__(self, model,
                 condition: str,
                 target_info: str = None,
                 mode: int = 2, 
                 rate: int = 10000, g_mode='+', device=0):
        """
        :param model: A SDM model.
        :param condition: The condition for computing the semantic loss.
        :param target_info: The target textural for textural loss.
        :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused
        :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
        """
        super().__init__()
        self.model = model
        self.condition = condition
        self.fn = nn.MSELoss(reduction="sum")
        self.target_info = target_info
        self.mode = mode
        self.rate = rate
        self.g_mode = g_mode
        
        print('g_mode:',  g_mode)

    def get_components(self, x):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """

        z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).to(x.device)
        c = self.model.get_learned_conditioning(self.condition)
        loss = self.model(z, c)[0]
        return z, loss

    def forward(self, x, components=False):
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describles the semantic content of the image.
        :return: The loss used for updating gradient in the adversarial attack.
        """
        g_dir = 1. if self.g_mode == '+' else -1.

        
        zx, loss_semantic = self.get_components(x)
        zy, loss_semantic_y = self.get_components(self.target_info)
        
        if components:
            return self.fn(zx, zy), loss_semantic
        if self.mode == 'advdm': # 
            return - loss_semantic * g_dir
        elif self.mode == 'texture_only':
            return self.fn(zx, zy)
        elif self.mode == 'mist':
            return self.fn(zx, zy) * g_dir  - loss_semantic * self.rate
        elif self.mode == 'texture_self_enhance':
            return - self.fn(zx, zy)
        else:
            raise KeyError('mode not defined')

class target_model_refnet(nn.Module):
    """
    A virtual model which computes the semantic and textural loss in forward function.
    """

    def __init__(self, model,
                 condition: str,
                 target_info: str = None,
                 mode: int = 2, 
                 rate: int = 10000, g_mode='+', device=0):
        """
        :param model: A SDM model.
        :param condition: The condition for computing the semantic loss.
        :param target_info: The target textural for textural loss.
        :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused
        :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
        """
        super().__init__()
        self.model = model
        self.condition = condition
        # self.fn = nn.MSELoss(reduction="sum")
        self.fn = nn.MSELoss(reduction="mean")
        self.target_info = target_info
        self.mode = mode
        self.rate = rate
        self.g_mode = g_mode
        
        print('g_mode:',  g_mode)

    def get_components(self, x, emb, return_latent=False):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """

        # z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).to(x.device)
        # c = self.model.get_learned_conditioning(self.condition)
        # loss = self.model(z, c)[0]
        # return z, loss

        reference_output = self.model.get_reference_output(x, emb, return_latent=return_latent)

        return reference_output

    def forward(self, x, components=False):
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describles the semantic content of the image.
        :return: The loss used for updating gradient in the adversarial attack.
        """
        g_dir = 1. if self.g_mode == '+' else -1.

        return_latent = False
        if self.g_mode=='latent':
            return_latent = True

        zx = self.get_components(x['image'], x['face_emb'], return_latent=return_latent)
        with torch.no_grad():
            zy = self.get_components(self.target_info['image'], self.target_info['face_emb'], return_latent=return_latent)

        if components:
            return self.fn(zx, zy), loss_semantic
        if self.mode == 'advdm': # 
            return - loss_semantic * g_dir
        elif self.mode == 'texture_only':
            return self.fn(zx, zy)
        elif self.mode == 'mist':
            return self.fn(zx, zy) * g_dir  - loss_semantic * self.rate
        elif self.mode == 'texture_self_enhance':
            return - self.fn(zx, zy)
        elif self.mode == 'refnet':
            if return_latent:
                return self.fn(zx, zy)
            else:
                return self.fn(zx[0], zy[0]) # tuple
        else:
            raise KeyError('mode not defined')

    # def forward(self, x, emb, components=False):
    #     """
    #     Compute the loss based on different mode.
    #     The textural loss shows the distance between the input image and target image in latent space.
    #     The semantic loss describles the semantic content of the image.
    #     :return: The loss used for updating gradient in the adversarial attack.
    #     """
    #     g_dir = 1. if self.g_mode == '+' else -1.

        
    #     zx, loss_semantic = self.get_components(x, emb)
    #     zy, loss_semantic_y = self.get_components(self.target_info)
        
    #     if components:
    #         return self.fn(zx, zy), loss_semantic
    #     if self.mode == 'advdm': # 
    #         return - loss_semantic * g_dir
    #     elif self.mode == 'texture_only':
    #         return self.fn(zx, zy)
    #     elif self.mode == 'mist':
    #         return self.fn(zx, zy) * g_dir  - loss_semantic * self.rate
    #     elif self.mode == 'texture_self_enhance':
    #         return - self.fn(zx, zy)
    #     else:
    #         raise KeyError('mode not defined')

class target_model_hallo(nn.Module):
    """
    A virtual model which computes the semantic and textural loss in forward function.
    """

    def __init__(self, model,
                 condition: str,
                 target_info: str = None,
                 mode: int = 2, 
                 rate: int = 10000, g_mode='+', device=0):
        """
        :param model: A SDM model.
        :param condition: The condition for computing the semantic loss.
        :param target_info: The target textural for textural loss.
        :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused
        :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
        """
        super().__init__()
        self.model = model
        self.condition = condition
        # self.fn = nn.MSELoss(reduction="sum")
        self.fn = nn.MSELoss(reduction="mean")
        self.target_info = target_info
        self.mode = mode
        self.rate = rate
        self.g_mode = g_mode
        
        print('g_mode:',  g_mode)

    def get_components(self, x):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """

        # z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).to(x.device)
        # c = self.model.get_learned_conditioning(self.condition)
        loss = self.model(x)
        # return z, loss
        return loss

    def get_reference_output(self, x, emb):

        reference_output = self.model.get_reference_output(x, emb)

        return reference_output

    def forward(self, x, components=False):
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describles the semantic content of the image.
        :return: The loss used for updating gradient in the adversarial attack.
        """
        g_dir = 1. if self.g_mode == '+' else -1.

        
        loss_semantic = self.get_components(x)
        # with torch.no_grad():
        #     zy = self.get_components(self.target_info['image'], self.target_info['face_emb'])

        if components:
            return self.fn(zx, zy), loss_semantic
        if self.mode == 'advdm': # 
            return - loss_semantic * g_dir
        elif self.mode == 'texture_only':
            return self.fn(zx, zy)
        elif self.mode == 'mist':
            return self.fn(zx, zy) * g_dir  - loss_semantic * self.rate
        elif self.mode == 'texture_self_enhance':
            return - self.fn(zx, zy)
        elif self.mode == 'refnet':
            return self.fn(zx[0], zy[0]) # tuple
        elif self.mode == 'hallo':
            return - loss_semantic * g_dir
        else:
            raise KeyError('mode not defined')


def init(epsilon: int = 16, steps: int = 100, alpha: int = 1, 
         input_size: int = 512, object: bool = False, seed: int =23, 
         ckpt: str = None, base: str = None, mode: int = 2, rate: int = 10000, g_mode='+', device=0, input_prompt='a photo', min_ts=0):
    """
    Prepare the config and the model used for generating adversarial examples.
    :param epsilon: Strength of adversarial attack in l_{\infinity}.
                    After the round and the clip process during adversarial attack, 
                    the final perturbation budget will be (epsilon+1)/255.
    :param steps: Iterations of the attack.
    :param alpha: strength of the attack for each step. Measured in l_{\infinity}.
    :param input_size: Size of the input image.
    :param object: Set True if the targeted images describes a specifc object instead of a style.
    :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused. 
                 See the document for more details about the mode.
    :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
    :returns: a dictionary containing model and config.
    """
    if mode == 'refnet':
        print(' Define refnet model...')
        model = HalloRefNet('./hallo/configs/train/stage2.yaml')
        net = target_model_refnet(model, input_prompt, mode=mode, rate=rate, g_mode=g_mode)

    elif mode == 'hallo':
        print(' Define hallo model...')
        model = HalloNet('./hallo/configs/train/stage2.yaml', min_timesteps=min_ts)
        net = target_model_hallo(model, input_prompt, mode=mode, rate=rate, g_mode=g_mode)

    else:
        print('unused currently.')
        assert(0)
        if ckpt is None:
            ckpt = '/home/gy/code/protect/Adversarial_Content_Attack/ckpt/model.ckpt'

        if base is None:
            base = '/home/gy/code/talking-head/hallo/protect/configs/stable-diffusion/v1-inference-attack.yaml'

        imagenet_templates_small_style = ['a painting']
        
        imagenet_templates_small_object = ['a photo']

        config_path = os.path.join(os.getcwd(), base)
        config = OmegaConf.load(config_path)

        ckpt_path = os.path.join(os.getcwd(), ckpt)

        model = load_model_from_config(config, ckpt_path, device=device).to(device)
        net = target_model(model, input_prompt, mode=mode, rate=rate, g_mode=g_mode)

    fn = identity_loss()

    # if object:
    #     imagenet_templates_small = imagenet_templates_small_object
    # else:
    #     imagenet_templates_small = imagenet_templates_small_style

    # input_prompt = [imagenet_templates_small[0] for i in range(1)]
    
    net.eval()

    # parameter
    parameters = {
        'epsilon': epsilon/255.0 * (1-(-1)),
        'alpha': alpha/255.0 * (1-(-1)),
        'steps': steps,
        'input_size': input_size,
        'mode': mode,
        'rate': rate,
        'g_mode': g_mode
    }

    return {'net': net, 'fn': fn, 'parameters': parameters}


def infer(img: PIL.Image.Image, config, tar_img: PIL.Image.Image = None, diff_pgd=None, using_target=False, device=0) -> np.ndarray:
    """
    Process the input image and generate the misted image.
    :param img: The input image or the image block to be misted.
    :param config: config for the attack.
    :param img: The target image or the target block as the reference for the textural loss.
    :returns: A misted image.
    """

    net = config['net']
    fn = config['fn']
    parameters = config['parameters']
    mode = parameters['mode']
    epsilon = parameters["epsilon"]
    g_mode = parameters['g_mode']
    
    cprint(f'epsilon: {epsilon}', 'y')
    
    alpha = parameters["alpha"]
    steps = parameters["steps"]
    input_size = parameters["input_size"]
    rate = parameters["rate"]
    
    
    img = img.convert('RGB')
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    
        
    img = img[:, :, :3]
    if tar_img is not None:
        tar_img = np.array(tar_img).astype(np.float32) / 127.5 - 1.0
        tar_img = tar_img[:, :, :3]
    trans = transforms.Compose([transforms.ToTensor()])
    
    data_source = torch.zeros([1, 3, input_size, input_size]).to(device)
    data_source[0] = trans(img).to(device)
    target_info = torch.zeros([1, 3, input_size, input_size]).to(device)
    target_info[0] = trans(tar_img).to(device)
    net.target_info = target_info
    if mode == 'texture_self_enhance':
        net.target_info = data_source
    net.mode = mode
    net.rate = rate
    label = torch.zeros(data_source.shape).to(device)
    print(net(data_source, components=True))

    # Targeted PGD attack is applied.
    
    time_start_attack = time.time()

    if mode in ['advdm', 'texture_only', 'mist', 'texture_self_enhance']: # using raw PGD
        
        attack = LinfPGDAttack(net, fn, epsilon, steps, eps_iter=alpha, clip_min=-1.0, targeted=True)
        attack_output = attack.perturb(data_source, label)
    
    elif mode == 'sds': # apply SDS to speed up the PGD
        
        print('using sds')
        
        attack =  Linf_PGD(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        
        attack_output, loss_all = attack.pgd_sds(X=data_source, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_image=target_info, target_rate=rate)
        
        print(torch.abs(attack_output-data_source).max())
        # print(loss_all)
        # emp = [wandb.log({'adv_loss':loss_item}) for loss_item in loss_all]
    
    elif mode == 'sds_z':
        print('using sds')
        
        attack =  Linf_PGD(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        dm = net.model
        with torch.no_grad():
            z = dm.get_first_stage_encoding(dm.encode_first_stage(data_source)).to(device)

        attack_output, loss_all = attack.pgd_sds_latent(z=z, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_image=target_info, target_rate=rate)
    

    elif mode == 'none':
        attack_output = data_source
    
    print('Attack takes: ', time.time() - time_start_attack)

    
    
    
    editor = SDEdit(net=net)
    edit_one_step = editor.edit_list(attack_output, restep=None, t_list=[0.01, 0.05, 0.1, 0.2, 0.3])
    edit_multi_step   = editor.edit_list(attack_output, restep='ddim100')
    
    
    # print(net(attack_output, components=True))

    output = attack_output[0]
    save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
    grid_adv = grid_adv
    return grid_adv,  edit_one_step, edit_multi_step

def infer_refnet(source_img, config, tar_img, diff_pgd=None, using_target=False, device=0) -> np.ndarray:
    """
    Process the input image and generate the misted image.
    :param img: The input image or the image block to be misted.
    :param config: config for the attack.
    :param img: The target image or the target block as the reference for the textural loss.
    :returns: A misted image.
    """

    net = config['net']
    fn = config['fn']
    parameters = config['parameters']
    mode = parameters['mode']
    epsilon = parameters["epsilon"]
    g_mode = parameters['g_mode']

    cprint(f'epsilon: {epsilon}', 'y')
    
    alpha = parameters["alpha"]
    steps = parameters["steps"]
    input_size = parameters["input_size"]
    rate = parameters["rate"]
    
    
    data_source = source_img
    
    
    if tar_img is not None:
        target_info = tar_img
        target_info['image'] = target_info['image'].to(device)

    data_source['image'] = data_source['image'].to(device)
    # print(data_source['image'].shape) # 3, 512, 512
    net.target_info = target_info
    if mode == 'texture_self_enhance':
        target_info = source_img
        target_info['image'] = target_info['image'].to(device)
        net.target_info = target_info
        assert(0)

    net.mode = mode
    net.rate = rate
    label = torch.zeros(data_source['image'].shape).to(device)
    print(net(data_source))
    
    # Targeted PGD attack is applied.
    
    time_start_attack = time.time()

    if mode in ['advdm', 'texture_only', 'mist', 'texture_self_enhance']: # using raw PGD
        
        attack = LinfPGDAttack(net, fn, epsilon, steps, eps_iter=alpha, clip_min=-1.0, targeted=True)
        attack_output = attack.perturb(data_source, label)
    
    elif mode == 'sds': # apply SDS to speed up the PGD
        
        print('using sds')
        
        attack = Linf_PGD(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        
        attack_output, loss_all = attack.pgd_sds(X=data_source, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_image=target_info, target_rate=rate)
        
        print(torch.abs(attack_output-data_source).max())
        # print(loss_all)
        # emp = [wandb.log({'adv_loss':loss_item}) for loss_item in loss_all]
    
    elif mode == 'sds_z':
        print('using sds')
        
        attack =  Linf_PGD(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        dm = net.model
        with torch.no_grad():
            z = dm.get_first_stage_encoding(dm.encode_first_stage(data_source)).to(device)

        attack_output, loss_all = attack.pgd_sds_latent(z=z, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_image=target_info, target_rate=rate)
    
    elif mode == 'refnet':
        print('using referencenet...')

        attack = Linf_PGD_Hallo(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        attack_output = attack.pgd_sds_refnet(X=data_source, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_info=target_info, target_rate=rate)
        
        print(torch.abs(attack_output-data_source['image']).max())

    elif mode == 'hallo':
        print('attack hallo')
        assert(0)

    elif mode == 'none':
        attack_output = data_source
    
    print('Attack takes: ', time.time() - time_start_attack)

    
    
    
    # editor = SDEdit(net=net)
    # edit_one_step = editor.edit_list(attack_output, restep=None, t_list=[0.01, 0.05, 0.1, 0.2, 0.3])
    # edit_multi_step   = editor.edit_list(attack_output, restep='ddim100')
    
    
    # print(net(attack_output, components=True))

    output = attack_output
    save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
    grid_adv = grid_adv
    return grid_adv #,  edit_one_step, edit_multi_step

def infer_hallo(source_img, config, tar_img, diff_pgd=None, using_target=False, device=0) -> np.ndarray:
    """
    Process the input image and generate the misted image.
    :param img: The input image or the image block to be misted.
    :param config: config for the attack.
    :param img: The target image or the target block as the reference for the textural loss.
    :returns: A misted image.
    """

    net = config['net']
    fn = config['fn']
    parameters = config['parameters']
    mode = parameters['mode']
    epsilon = parameters["epsilon"]
    g_mode = parameters['g_mode']

    cprint(f'epsilon: {epsilon}', 'y')
    
    alpha = parameters["alpha"]
    steps = parameters["steps"]
    input_size = parameters["input_size"]
    rate = parameters["rate"]
    
    
    data_source = source_img
    
    
    if tar_img is not None:
        target_info = tar_img
        target_info['image'] = target_info['image'].to(device)

    data_source['image'] = data_source['image'].to(device)
    # print(data_source['image'].shape) # 3, 512, 512
    net.target_info = target_info
    if mode == 'texture_self_enhance':
        target_info = source_img
        target_info['image'] = target_info['image'].to(device)
        net.target_info = target_info
        assert(0)

    net.mode = mode
    net.rate = rate
    label = torch.zeros(data_source['image'].shape).to(device)
    
    # Targeted PGD attack is applied.
    
    time_start_attack = time.time()

    if mode in ['advdm', 'texture_only', 'mist', 'texture_self_enhance']: # using raw PGD
        
        attack = LinfPGDAttack(net, fn, epsilon, steps, eps_iter=alpha, clip_min=-1.0, targeted=True)
        attack_output = attack.perturb(data_source, label)
    
    elif mode == 'sds': # apply SDS to speed up the PGD
        
        print('using sds')
        
        attack = Linf_PGD(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        
        attack_output, loss_all = attack.pgd_sds(X=data_source, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_image=target_info, target_rate=rate)
        
        print(torch.abs(attack_output-data_source).max())
        # print(loss_all)
        # emp = [wandb.log({'adv_loss':loss_item}) for loss_item in loss_all]
    
    elif mode == 'sds_z':
        print('using sds')
        
        attack =  Linf_PGD(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        dm = net.model
        with torch.no_grad():
            z = dm.get_first_stage_encoding(dm.encode_first_stage(data_source)).to(device)

        attack_output, loss_all = attack.pgd_sds_latent(z=z, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_image=target_info, target_rate=rate)
    
    elif mode == 'refnet':
        print('using referencenet...')

        attack = Linf_PGD_Hallo(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        attack_output = attack.pgd_sds_refnet(X=data_source, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_info=target_info, target_rate=rate)
        
        print(torch.abs(attack_output-data_source['image']).max())

    elif mode == 'hallo':
        print('attack hallo')
        attack = Linf_PGD_Hallo(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        attack_output = attack.pgd_sds_hallo(X=data_source, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_info=target_info, target_rate=rate)
        
        print(torch.abs(attack_output-data_source['image']).max())

    elif mode == 'none':
        attack_output = data_source
    
    print('Attack takes: ', time.time() - time_start_attack)
    
    # editor = SDEdit(net=net)
    # edit_one_step = editor.edit_list(attack_output, restep=None, t_list=[0.01, 0.05, 0.1, 0.2, 0.3])
    # edit_multi_step   = editor.edit_list(attack_output, restep='ddim100')
    
    
    # print(net(attack_output, components=True))

    output = attack_output
    save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
    grid_adv = grid_adv
    return grid_adv #,  edit_one_step, edit_multi_step


# Test the script with command: python mist_v2.py 16 100 512 1 2 1
# or the command: python mist_v2.py 16 100 512 2 2 1, which process
# the image blockwisely for lower VRAM cost


@hydra.main(version_base=None, config_path="./configs/attack", config_name="base")
def main(cfg : DictConfig):
    print(cfg.attack)
    time_start = time.time()
    
    args = cfg.attack
    
    
    epsilon = args.epsilon
    steps = args.steps
    input_size = args.input_size
    mode = args.mode
    alpha = args.alpha
    rate = args.target_rate if not mode == 'mist' else 1e4
    g_mode = args.g_mode
    output_path, img_path = args.output_path, args.img_path
    diff_pgd = args.diff_pgd
    using_target = args.using_target
    device = args.device
    min_ts = args.min_timesteps
    
    if using_target and mode == 'sds':
        mode_name = f'{mode}T{rate}'
    else:
        mode_name = mode

    output_path = output_path + f'/{mode_name}_eps{epsilon}_steps{steps}_gmode{g_mode}_{min_ts}'
    if diff_pgd[0]:
        output_path = output_path + '_diffpgd/'
    else:
        output_path += '/'
    
    mp(output_path)
    
    input_prompt = 'a photo'
    if 'anime' in img_path:
        input_prompt = 'an anime picture'
    elif 'artwork' in img_path:
        input_prompt = 'an artwork painting'
    elif 'landscape' in img_path:
        input_prompt = 'a landscape photo'
    else:
        input_prompt = 'a portrait photo'
        # input_prompt = 'a photo'
    
    
    
    
    config = init(epsilon=epsilon, alpha=alpha, steps=steps, 
                  mode=mode, rate=rate, g_mode=g_mode, device=device, 
                  input_prompt=input_prompt, min_ts=min_ts)

    img_paths = glob.glob(img_path+'/*.png') + glob.glob(img_path+'/*.jpg') + glob.glob(img_path+'/*.jpeg')
    # img_paths.sort(key=lambda x: int(x[x.rfind('/')+1:x.rfind('.')]))
    img_path = img_path[:args.max_exp_num]



    if mode == 'refnet':

        print('2. prepare data')
        img_size = (512, 512)
        face_analysis_model_path = './pretrained_models/face_analysis'
        save_path = './.cache'
        face_expand_ratio = 1.2
        ###  one image everytime. TODO: process more images
        source_image_paths = glob.glob(img_path+'/*.png') + glob.glob(img_path+'/*.jpg') + glob.glob(img_path+'/*.jpeg')
        target_image_processor = ImageProcessor(img_size, face_analysis_model_path)
        target_image_path = 'protect/test_images/target/MIST.png'
        target_image_pixels, target_image_face_emb = target_image_processor.preprocess_target_img(
                target_image_path, save_path, face_expand_ratio)
        target_input = {'image': target_image_pixels, 'face_emb': target_image_face_emb}
        for source_image_path in source_image_paths:
            cprint(f'Processing: [{source_image_path}]', 'y')

            rsplit_image_path = source_image_path.rsplit('/')
            file_name = f"/{rsplit_image_path[-2]}/{rsplit_image_path[-1]}/"
            file_name = file_name.rsplit('.')[0]
            mp(output_path + file_name)
            with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
                source_image_pixels, \
                source_image_face_region, \
                source_image_face_emb, \
                source_image_full_mask, \
                source_image_face_mask, \
                source_image_lip_mask = image_processor.preprocess(
                    source_image_path, save_path, face_expand_ratio)
            source_input = {'image': source_image_pixels, 'face_emb': source_image_face_emb}
            output_image = infer_refnet(source_input, config, target_input, diff_pgd=diff_pgd, using_target=using_target, device=device)
            
            output = Image.fromarray(output_image.astype(np.uint8))
            time_start_sdedit = time.time()
            
            output_name = output_path + f'/{file_name}.png'
            
            output.save(output_name)
            
            print('TIME CMD=', time.time() - time_start)

    elif mode == 'hallo':
        print('2. prepare data')
        img_size = (512, 512)
        face_analysis_model_path = './pretrained_models/face_analysis'
        save_path = './.cache'
        face_expand_ratio = 1.2
        ###  one image everytime. TODO: process more images
        source_image_paths = glob.glob(img_path+'/*.png') + glob.glob(img_path+'/*.jpg') + glob.glob(img_path+'/*.jpeg')
        image_processor = ImageProcessor(img_size, face_analysis_model_path)
        target_image_path = 'protect/test_images/target/MIST.png'
        target_image_pixels, target_image_face_emb = image_processor.preprocess_target_img(
                target_image_path, save_path, face_expand_ratio)
        target_input = {'image': target_image_pixels, 'face_emb': target_image_face_emb}
        for source_image_path in tqdm(source_image_paths):
            cprint(f'Processing: [{source_image_path}]', 'y')

            rsplit_image_path = source_image_path.rsplit('/')
            file_name = f"/{rsplit_image_path[-2]}/{rsplit_image_path[-1]}/"
            file_name = file_name.rsplit('.')[0]
            mp(output_path + file_name)
            source_image_pixels, \
            source_image_face_region, \
            source_image_face_emb, \
            source_image_full_mask, \
            source_image_face_mask, \
            source_image_lip_mask = image_processor.preprocess(
                source_image_path, save_path, face_expand_ratio)

            source_input = {'image':        source_image_pixels,
                            'face_region':  source_image_face_region,
                            'face_emb':     torch.from_numpy(source_image_face_emb),
                            'full_mask':    source_image_full_mask,
                            'face_mask':    source_image_face_mask,
                            'lip_mask':     source_image_lip_mask
                            }
            output_image = infer_hallo(source_input, config, target_input, diff_pgd=diff_pgd, using_target=using_target, device=device)
            
            output = Image.fromarray(output_image.astype(np.uint8))
            time_start_sdedit = time.time()
            
            output_name = output_path + f'/{file_name}.png'
            
            output.save(output_name)
            
            print('TIME CMD=', time.time() - time_start)

    else:
        for image_path in tqdm(img_paths):
            cprint(f'Processing: [{image_path}]', 'y')
    
            rsplit_image_path = image_path.rsplit('/')
            file_name = f"/{rsplit_image_path[-2]}/{rsplit_image_path[-1]}/"
            file_name = file_name.rsplit('.')[0]
            mp(output_path + file_name)
            
            target_image_path = 'test_images/target/MIST.png'
            img = load_image_from_path(image_path, input_size)
            tar_img = load_image_from_path(target_image_path, input_size)
            
            
            bls = input_size//1
            config['parameters']["input_size"] = bls

            output_image = np.zeros([input_size, input_size, 3])
            
            
            for i in tqdm(range(1)):
                for j in tqdm(range(1)):
                    img_block = Image.fromarray(np.array(img)[bls*i: bls*i+bls, bls*j: bls*j + bls])
                    tar_block = Image.fromarray(np.array(tar_img)[bls*i: bls*i+bls, bls*j: bls*j + bls])
                    output_image[bls*i: bls*i+bls, bls*j: bls*j + bls], edit_one_step, edit_multi_step = infer(img_block, config, tar_block, diff_pgd=diff_pgd, using_target=using_target, device=device)
            
            output = Image.fromarray(output_image.astype(np.uint8))
            
            time_start_sdedit = time.time()
            si(edit_one_step, output_path + f'{file_name}_onestep.png')
            si(edit_multi_step, output_path + f'{file_name}_multistep.png')
            print('SDEdit takes: ', time.time() - time_start_sdedit)
            
            
            output_name = output_path + f'/{file_name}_attacked.png'
            
            output.save(output_name)
            
            
            print('TIME CMD=', time.time() - time_start)


if __name__ == '__main__':
    main()
