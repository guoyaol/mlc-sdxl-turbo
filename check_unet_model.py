from diffusers import DiffusionPipeline
from web_stable_diffusion import utils
import torch

device_str = "cpu"
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")

unet = utils.get_unet(pipe, device_str)
unet.eval()


input1 = torch.rand((1, 4, 64, 64)).to(torch.float32)
input2 = torch.tensor(3).to(torch.int32)
input3 = torch.rand((1, 77, 2048)).to(torch.float32)
input4 = torch.rand((1, 1280)).to(torch.float32)
input5 = torch.rand((1, 6)).to(torch.float32)

class UNetModelWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        # Default guidance scale factor in stable diffusion.
        self.guidance_scale = 5.0

    def forward(self, latents, timestep_tensor, text_embeddings, added_cond_kwargs_text_embeds, added_cond_kwargs_text_time_ids):
        # Latent concatenation.
        latent_model_input = torch.cat([latents] * 2, dim=0)
        # UNet forward.
        noise_pred = self.unet(latent_model_input, timestep_tensor, text_embeddings, added_cond_kwargs_text_embeds, added_cond_kwargs_text_time_ids)
        # Classifier-free guidance.
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred
    
class ref_UNetModelWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        # Default guidance scale factor in stable diffusion.
        self.guidance_scale = 5.0

    def forward(self, latents, timestep_tensor, text_embeddings, added_dict):
        # Latent concatenation.
        latent_model_input = torch.cat([latents] * 2, dim=0)
        # UNet forward.
        noise_pred = self.unet(latent_model_input, timestep_tensor, text_embeddings, added_cond_kwargs = added_dict, return_dict=False)[0]
        # Classifier-free guidance.
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred

print("our result")

# our_unt_noise = UNetModelWrapper(unet)

with torch.no_grad():
    our_result = unet(input1, input2, input3, input4, input5)
print(our_result)


added_cond_kwargs = {"text_embeds": input4, "time_ids": input5}
# noise_pred = self.unet(
#     latent_model_input,
#     t,
#     encoder_hidden_states=prompt_embeds,
#     cross_attention_kwargs=cross_attention_kwargs,
#     added_cond_kwargs=added_cond_kwargs,
#     return_dict=False,
# )[0]
pipe.unet.eval()
# ref_unet_noise = ref_UNetModelWrapper(pipe.unet)
with torch.no_grad():
    ref_result = pipe.unet(input1, input2, input3, added_cond_kwargs = added_cond_kwargs, return_dict=False)[0]

print("ref result")
print(ref_result)



print(our_result.shape)
print(ref_result.shape)
import numpy as np
np.testing.assert_allclose(our_result.numpy(), ref_result.numpy(), atol=1e-5)
print("model check success")