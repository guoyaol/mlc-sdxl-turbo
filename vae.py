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

def vae_to_image(pipe) -> tvm.IRModule:
    class VAEModelWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            # Scale the latents so that it can be decoded by VAE.
            latents = 1 / 0.13025 * latents
            # VAE decode
            # z = self.vae.post_quant_conv(latents)
            image = self.vae.decode(latents, return_dict=False)[0]
            # Image normalization
            image = (image / 2 + 0.5).clamp(min=0, max=1)
            image = (image.permute(0, 2, 3, 1) * 255).round()
            return image

    vae = utils.get_vae(pipe)
    vae_to_image = VAEModelWrapper(vae)

    graph = fx.symbolic_trace(vae_to_image)
    mod = from_fx(
        graph,
        [((1, 4, 64, 64), "float32")],
        keep_params_as_input=True,
    )
    return tvm.IRModule({"vae": mod["main"]})

pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
vae = vae_to_image(pipe)

print("successful import")
