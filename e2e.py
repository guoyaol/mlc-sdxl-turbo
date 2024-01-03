import tvm
from web_stable_diffusion import utils
from tvm import relax
import torch

# Load the model weight parameters back.
target = tvm.target.Target("apple/m2-gpu")
device = tvm.metal()
torch_device = "mps"

const_params_dict = utils.load_params(artifact_path="dist", device=device)
# Load the model executable back from the shared library.
ex = tvm.runtime.load_module("dist/stable_diffusion.so")

vm = relax.VirtualMachine(rt_mod=ex, device=device)



import json
import numpy as np

from web_stable_diffusion import runtime

from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer


class TVMSDXLTurboPipeline:
    def __init__(
        self,
        vm: relax.VirtualMachine,
        tokenizer: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        scheduler: runtime.Scheduler,
        tvm_device,
        param_dict,
    ):
        def wrapper(f, params):
            def wrapped_f(*args):
                return f(*args, params)

            return wrapped_f

        self.vm = vm
        self.clip_to_text_embeddings = wrapper(vm["clip"], param_dict["clip"])
        self.clip_to_text_embeddings2 = wrapper(vm["clip2"], param_dict["clip2"])
        self.unet_latents_to_noise_pred = wrapper(vm["unet"], param_dict["unet"])
        self.vae_to_image = wrapper(vm["vae"], param_dict["vae"])
        self.concat_embeddings = vm["concat_embeddings"]
        self.concat_pool_embeddings = vm["concat_pool_embeddings"]
        self.concat_enocder_outputs = vm["concat_enocder_outputs"]
        self.image_to_rgba = vm["image_to_rgba"]
        self.cat_latents = vm["cat_latents"]
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.scheduler = scheduler
        self.tvm_device = tvm_device
        self.param_dict = param_dict

    def __call__(self, prompt: str, negative_prompt: str = ""):
        # The height and width are fixed to 512.

        # Compute the embeddings for the prompt and negative prompt.
        list_text_embeddings = []

        tokenizers = [self.tokenizer, self.tokenizer2]
        text_encoders = [self.clip_to_text_embeddings, self.clip_to_text_embeddings2]

        prompt_embeds_list = []

        #prompt
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            # TODO: better if can find a better tokenizer
            text_input_ids = text_inputs.input_ids.to(torch.int32)


            # Clip the text if the length exceeds the maximum allowed length.
            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
  

            # Compute text embeddings.
            text_input_ids = tvm.nd.array(text_input_ids.cpu().numpy(), self.tvm_device)
            clip_output = text_encoder(text_input_ids)
            text_embeddings = clip_output[0]
            pooled_prompt_embeds = clip_output[1]

            prompt_embeds_list.append(text_embeddings)
        
        prompt_embeds = self.concat_enocder_outputs(prompt_embeds_list[0], prompt_embeds_list[1])

        # if negative_prompt != "":
        neg_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            neg_text_inputs = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

            neg_text_input_ids = neg_text_inputs.input_ids.to(torch.int32)


            if neg_text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                neg_text_input_ids = neg_text_input_ids[:, : self.tokenizer.model_max_length]
            neg_text_input_ids = tvm.nd.array(neg_text_input_ids.cpu().numpy(), self.tvm_device)
            neg_clip_output = text_encoder(neg_text_input_ids)
            neg_text_embeddings = neg_clip_output[0]
            neg_pooled_prompt_embeds = neg_clip_output[1]
            neg_prompt_embeds_list.append(neg_text_embeddings)

        neg_prompt_embeds = self.concat_enocder_outputs(neg_prompt_embeds_list[0], neg_prompt_embeds_list[1])
        # else:
        #     torch_template = torch.from_numpy(prompt_embeds.asnumpy())
        #     neg_prompt_embeds = torch.zeros_like(torch_template)
        #     neg_prompt_embeds = tvm.nd.array(neg_prompt_embeds, self.tvm_device)

        #     torch_template_pooled = torch.from_numpy(pooled_prompt_embeds.asnumpy())
        #     neg_pooled_prompt_embeds = torch.zeros_like(torch_template_pooled)
        #     neg_pooled_prompt_embeds = tvm.nd.array(neg_pooled_prompt_embeds, self.tvm_device)


            
        add_text_embeds = pooled_prompt_embeds
        input_text_embeddings = prompt_embeds

        print("add_text_embeds", add_text_embeds)
        print("input_text_embeddings", input_text_embeddings)

        add_time_ids = torch.tensor([[512., 512.,   0.,   0., 512., 512.]], dtype=torch.float32)
        add_time_ids = tvm.nd.array(add_time_ids, self.tvm_device)


        # Randomly initialize the latents.
        torch.manual_seed(42)
        # latents = torch.randn(
        #     (1, 4, 128, 128),
        #     device="cpu",
        #     dtype=torch.float32,
        # )
        latents = torch.randn((1, 4, 64, 64), generator=None, device=torch_device, dtype=torch.float32, layout=torch.strided)
        latents = latents.cpu()

        #TODO: change to init noise sigma of scheduler
        latents = 14.6146 * latents
        latents = tvm.nd.array(latents.numpy(), self.tvm_device)

        print("initialized latents", latents)

        # UNet iteration.
        for i in tqdm(range(len(self.scheduler.timesteps))):
            #TODO: implement scheduler runtime, sigma... all the things
            t = self.scheduler.timesteps[i]
            # latent_model_input = self.cat_latents(latents)
            latent_model_input = latents

            scaled_latent_model_input = self.scheduler.scale_model_input(self.vm, latent_model_input, i)

            noise_pred = self.unet_latents_to_noise_pred(scaled_latent_model_input, t, input_text_embeddings, add_text_embeds, add_time_ids)
            print("noise_pred shape: ", noise_pred)
            #TODO: add noise
            noise = torch.randn([1, 4, 64, 64], dtype=torch.float32, layout=torch.strided)
            noise = tvm.nd.array(noise.numpy(), self.tvm_device)
            latents = self.scheduler.step(self.vm, noise_pred, latents, i, noise)
            print("latents", latents)

        # VAE decode.
        image = self.vae_to_image(latents)

        # Transform generated image to RGBA mode.
        image = self.image_to_rgba(image)
        return Image.fromarray(image.numpy().view("uint8").reshape(512, 512, 4))


pipe = TVMSDXLTurboPipeline(
    vm=vm,
    tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
    tokenizer2=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", pad_token = "!"),
    scheduler=runtime.EulerAncestralDiscreteScheduler(artifact_path="dist", device=device),
    tvm_device=device,
    param_dict=const_params_dict,
)


import time

prompt = "Jellyfish floating in a forest"
negative_prompt = "purple"

start = time.time()
image = pipe(prompt, negative_prompt)
end = time.time()

image.save('jellyfish.png')

print(f"Time elapsed: {end - start} seconds.")