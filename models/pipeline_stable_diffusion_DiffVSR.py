# Copyright 2023 The HuggingFace Team. All rights reserved.
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
# limitations under the License.

import inspect
from typing import Any, Callable, List, Optional, Union

import numpy as np
import math
import random
import PIL
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from .scheduling_ddim import DDIMScheduler

from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDPMScheduler

from diffusers.utils import deprecate, is_accelerate_available, is_accelerate_version, logging, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


from einops import rearrange


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 64 for x in (w, h))  # resize to integer multiple of 64

        image = [np.array(i.resize((w, h)))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def fuse_latents(latents_list, short_seq, stride, output_seq):
    B, C, _, H, W = latents_list[0].shape

    # Initialize final fusion tensor
    final_latents = torch.zeros((B, C, output_seq, H, W), device=latents_list[0].device)

    for i, latents in enumerate(latents_list):
        start_idx = i * stride
        end_idx = start_idx + latents.shape[2]

        if i == 0:
            # Copy first part directly
            final_latents[:, :, start_idx:end_idx, :, :] = latents
        else:
            # Calculate overlap size
            overlap_size = short_seq - stride
            non_overlap_start = start_idx + overlap_size

            # Weighted fusion for overlapping parts
            for j in range(overlap_size):
                weight1 = 0.2 + 0.6 * (1 - j / (overlap_size - 1))
                weight2 = 0.2 + 0.6 * (j / (overlap_size - 1))
                final_latents[:, :, start_idx + j, :, :] = (weight1 * latents_list[i - 1][:, :, stride + j, :, :] +
                                                            weight2 * latents[:, :, j, :, :])

            # Copy non-overlapping parts directly
            remaining_frames = latents.shape[2] - overlap_size
            final_latents[:, :, non_overlap_start:non_overlap_start + remaining_frames, :, :] = latents[:, :, overlap_size:, :, :]

    return final_latents


class StableDiffusionUpscalePipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    _optional_components = ["feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        low_res_scheduler: DDPMScheduler,
        scheduler: DDIMScheduler,
        feature_extractor: Optional[CLIPImageProcessor] = None,
        max_noise_level: int = 350,
    ):
        super().__init__()

        if hasattr(vae, "config"):
            is_vae_scaling_factor_set_to_0_08333 = (hasattr(vae.config, "scaling_factor") 
                                                   and vae.config.scaling_factor == 0.08333)
            if not is_vae_scaling_factor_set_to_0_08333:
                deprecation_message = (
                    "The configuration file of the vae does not contain `scaling_factor` or it is set to"
                    f" {vae.config.scaling_factor}. If your checkpoint is a fine-tuned"
                    " version of `stabilityai/stable-diffusion-x4-upscaler` you should change 'scaling_factor' to"
                    " 0.08333. Please make sure to update the config accordingly as leaving it as is might lead to"
                    " incorrect results in future versions. If you have downloaded this checkpoint from the Hugging"
                    " Face Hub, it would be very nice if you could open a Pull Request for the `vae/config.json`"
                    " file"
                )
                deprecate("wrong scaling_factor", "1.0.0", deprecation_message, standard_warn=False)
                vae.register_to_config(scaling_factor=0.08333)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(max_noise_level=max_noise_level)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        """
        Offloads all models to CPU using accelerate to significantly reduce memory usage. 
        Models are moved to a torch.device('meta') and loaded to GPU only when needed.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        """
        Offloads all models to CPU using accelerate. Compared to sequential offload,
        this moves one whole model at a time to GPU when needed. Better performance but less memory savings.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()

        hook = None
        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        """
        Returns the device on which the pipeline's models will be executed.
        After CPU offload, device can only be inferred from Accelerate's module hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (hasattr(module, "_hf_hook") and 
                hasattr(module._hf_hook, "execution_device") and 
                module._hf_hook.execution_device is not None):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.
        
        Args:
            prompt: Text prompt to encode
            device: Target device
            num_images_per_prompt: Number of images per prompt
            do_classifier_free_guidance: Whether to use classifier free guidance
            negative_prompt: The prompt not to guide image generation
            prompt_embeds: Pre-generated text embeddings
            negative_prompt_embeds: Pre-generated negative text embeddings
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # Process multi-vector tokens if needed
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}")

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # Duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # Get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type as `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}")
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")
            else:
                uncond_tokens = negative_prompt

            # Process multi-vector tokens if needed
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # Duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def prepare_extra_step_kwargs(self, generator, eta):
        """
        Prepare extra kwargs for the scheduler step.
        eta (η) is only used with DDIMScheduler, ignored for other schedulers.
        eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        """
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def decode_latents(self, latents):
        """
        Decode the latents to image.
        """
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def decode_latents_vsr(self, latents, short_seq=None):
        """
        Decode the latents for video super resolution.
        """
        latents = 1 / self.vae.config.scaling_factor * latents
        if short_seq is not None:
            image = self.vae.decode(latents, num_frames=short_seq).sample
        else:
            image = self.vae.decode(latents).sample
        image = image.clamp(-1, 1).cpu()
        return image

    def check_inputs(
        self,
        prompt,
        image,
        noise_level,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        """
        Validate input parameters.
        
        Args:
            prompt: Text prompt
            image: Input image
            noise_level: Noise level to add
            callback_steps: Number of steps between callbacks
            negative_prompt: Negative text prompt
            prompt_embeds: Pre-generated text embeddings
            negative_prompt_embeds: Pre-generated negative text embeddings
        """
        if (callback_steps is None) or (
            callback_steps is not None and not isinstance(callback_steps, int) or callback_steps <= 0
        ):
            raise ValueError(
                f"`callback_steps` must be a positive integer but is {callback_steps} of type {type(callback_steps)}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot provide both `prompt` and `prompt_embeds`")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Must provide either `prompt` or `prompt_embeds`")
        elif prompt is not None and not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError("`prompt` must be of type `str` or `list`")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot provide both `negative_prompt` and `negative_prompt_embeds`")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape"
                )

        if not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image) and not isinstance(image, list):
            raise ValueError("`image` must be of type `torch.Tensor`, `PIL.Image.Image` or `list`")

        # Verify batch size of prompt and image
        if isinstance(image, list) or isinstance(image, torch.Tensor):
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            image_batch_size = len(image) if isinstance(image, list) else image.shape[0]
            if batch_size != image_batch_size:
                raise ValueError(f"Batch size mismatch: prompt ({batch_size}) != image ({image_batch_size})")

        if noise_level > self.config.max_noise_level:
            raise ValueError(f"`noise_level` must be <= {self.config.max_noise_level}")

    def prepare_latents_3d(
        self,
        batch_size,
        num_channels_latents,
        seq_len,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None
    ):
        """
        Prepare latent vectors for 3D generation.
        
        Args:
            batch_size: Number of samples in batch
            num_channels_latents: Number of channels in latents
            seq_len: Sequence length
            height: Height of latents
            width: Width of latents
            dtype: Data type
            device: Target device
            generator: Random number generator
            latents: Optional pre-generated latents
        """
        shape = (batch_size, num_channels_latents, seq_len, height, width)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # Scale initial noise by scheduler's required standard deviation
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_timesteps(self, num_inference_steps, strength, device):
        """
        Get timesteps for inference.
        
        Args:
            num_inference_steps: Number of inference steps
            strength: Strength parameter for timestep selection
            device: Target device
        """
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents_inversion(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None
    ):
        """
        Prepare latents for inversion process.
        
        Args:
            image: Input image
            timestep: Current timestep
            batch_size: Number of samples in batch
            num_images_per_prompt: Number of images per prompt
            dtype: Data type
            device: Target device
            generator: Random number generator
        """
        image = image.to(device=device, dtype=dtype)
        batch_size = batch_size * num_images_per_prompt

        b = image.shape[0]
        image = rearrange(image, 'b c t h w -> (b t) c h w').contiguous()
        image = F.interpolate(image, scale_factor=4, mode='bicubic')
        image = image.to(dtype=torch.float32)
        init_latents = self.vae.encode(image).latent_dist.sample(generator)
        torch.cuda.empty_cache()
        init_latents = rearrange(init_latents, '(b t) c h w -> b c t h w', b=b).contiguous()

        init_latents = self.vae.config.scaling_factor * init_latents
        init_latents = init_latents.to(dtype=torch.float16)

        # Add noise
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        num_inference_steps: int = 75,
        guidance_scale: float = 9.0,
        noise_level: int = 20,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1
    ):
        """
        Main pipeline function for image generation and manipulation.
        
        Args:
            prompt: Text prompt for generation
            image: Input image to process
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            noise_level: Amount of noise to add
            negative_prompt: Text prompt to not guide generation
            num_images_per_prompt: Number of images to generate per prompt
            eta: DDIM eta parameter
            generator: Random number generator
            latents: Pre-generated latent vectors
            prompt_embeds: Pre-computed text embeddings
            negative_prompt_embeds: Pre-computed negative text embeddings
            return_dict: Whether to return a StableDiffusionPipelineOutput
            callback: Function called at each step
            callback_steps: Number of steps between callbacks
        """
        # 1. Check inputs
        self.check_inputs(
            prompt,
            image,
            noise_level,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Preprocess image on CPU to reduce GPU memory pressure (keep model dtype)
        target_dtype = prompt_embeds.dtype
        image = image.to(dtype=target_dtype, device="cpu")

        # 5. Add noise to image using a CPU generator clone (to keep determinism)
        base_noise_level = torch.tensor([noise_level], dtype=torch.long, device=image.device)

        def _clone_to_cpu_generator(gen):
            if gen is None:
                return None
            if isinstance(gen, list):
                return [_clone_to_cpu_generator(g) for g in gen]
            if gen.device.type == "cpu":
                return gen
            seed = gen.initial_seed()
            return torch.Generator(device="cpu").manual_seed(seed)

        cpu_generator = _clone_to_cpu_generator(generator)
        noise = randn_tensor(image.shape, generator=cpu_generator, device=image.device, dtype=target_dtype)
        image = self.low_res_scheduler.add_noise(image, noise, base_noise_level)

        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = torch.cat([image] * batch_multiplier * num_images_per_prompt)
        noise_level = torch.cat([base_noise_level] * image.shape[0])

        # 6. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables
        seq_len, height, width = image.shape[2:]
        num_channels_latents = self.vae.config.latent_channels
        latents_ori = self.prepare_latents_3d(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            seq_len,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents_ori = latents_ori.to("cpu")

        window_size = 8
        stride = 4
        
        # Noise strategy
        noise_strategy = "rearrange"  # Options: none, rearrange
        
        if noise_strategy == "rearrange":
            for frame_index in range(window_size, image.shape[2], stride):
                list_index = list(range(
                    frame_index - window_size,
                    frame_index + stride - window_size,
                ))
                random.shuffle(list_index)
                latents_ori[:, :, frame_index:frame_index + stride] = latents_ori[:, :, list_index]
            logger.info("Noise rearranged")
            
        elif noise_strategy == "none":
            # Keep original noise unchanged
            logger.info("Using original noise without modification")
            
            
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        vframes_seq = image.shape[2]
        short_seq = window_size

        image_cpu = image

        latents_chunks = []
        if vframes_seq > short_seq:
            chunk_starts = range(0, vframes_seq - short_seq + stride, stride)
        else:
            chunk_starts = [0]

        noise_level_device = noise_level.to(device)

        for start_f in chunk_starts:
            end_f = min(start_f + short_seq, vframes_seq)
            logger.info(f'Processing: [{start_f}-{end_f}/{vframes_seq}]')
            torch.cuda.empty_cache()

            latents = latents_ori[:, :, start_f:end_f].to(device, dtype=prompt_embeds.dtype)
            image_chunk = image_cpu[:, :, start_f:end_f].to(device, dtype=prompt_embeds.dtype)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    torch.cuda.empty_cache()
                    # Expand latents for classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    # Scale model input
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # Predict noise
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        image_chunk,
                        encoder_hidden_states=prompt_embeds,
                        class_labels=noise_level_device
                    ).sample

                    # Perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Compute previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # Call callback if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                                  (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

                    del latent_model_input, noise_pred

            latents_chunks.append(latents.to("cpu"))
            del latents, image_chunk
            torch.cuda.empty_cache()

        if len(latents_chunks) > 1:
            latents = fuse_latents(latents_chunks, short_seq, stride, vframes_seq)
        else:
            latents = latents_chunks[0]

        # 10. Post-processing
        self.vae.to(dtype=torch.float32)
        latents = latents.float()

        # 11. Convert to frames
        short_seq = 4
        latents = rearrange(latents, 'b c t h w -> (b t) c h w').contiguous()
        vae_device = next(self.vae.parameters()).device
        if latents.shape[0] > short_seq:  # For video super resolution
            image = []
            for start_f in range(0, latents.shape[0], short_seq):
                torch.cuda.empty_cache()
                end_f = min(latents.shape[0], start_f + short_seq)
                latents_chunk = latents[start_f:end_f].to(vae_device)
                image_ = self.decode_latents_vsr(latents_chunk, short_seq=short_seq)
                image.append(image_)
                del image_, latents_chunk
            image = torch.cat(image, dim=0)
        else:
            image = self.decode_latents_vsr(latents.to(vae_device), short_seq=short_seq)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
