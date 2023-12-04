from diffusers import DiffusionPipeline
from diffusers.utils import load_image

pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")

pipe.to("mps")

prompt = "an astronaut riding horse on mars"

image = pipe(prompt, num_inference_steps=2, guidance_scale=0.0).images[0]

image.save("image.png")