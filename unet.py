from diffusers import DiffusionPipeline
import torch


import tvm
from tvm import relax
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R
from web_stable_diffusion import utils

import torch
from torch import fx

print(tvm.__file__)

#TODO: support fp16
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")

def unet_latents_to_noise_pred(pipe, device_str: str) -> tvm.IRModule:
    class UNetModelWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
            # Default guidance scale factor in stable diffusion.
            self.guidance_scale = 0.0

        def forward(self, latents, timestep_tensor, text_embeddings, added_cond_kwargs_text_embeds, added_cond_kwargs_text_time_ids):
            # UNet forward.
            noise_pred = self.unet(latents, timestep_tensor, text_embeddings, added_cond_kwargs_text_embeds, added_cond_kwargs_text_time_ids)
            # Classifier-free guidance.
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            return noise_pred

    unet = utils.get_unet(pipe, device_str)
    unet_to_noise_pred = UNetModelWrapper(unet)
    graph = fx.symbolic_trace(unet_to_noise_pred)
    mod = from_fx(
        graph,
        [((1, 4, 64, 64), "float32"), ((), "int32"), ((1, 77, 2048), "float32"), 
         ((1, 1280), "float32"), ((1, 6), "float32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"unet": mod["main"]})

torch_dev_key = "cpu"
unet = unet_latents_to_noise_pred(pipe, torch_dev_key)

print("successfully import")

