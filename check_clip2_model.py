from diffusers import DiffusionPipeline
import torch


import tvm
from tvm import relax
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R

import torch
from torch import fx

from transformers import CLIPTokenizer
from diffusers import DiffusionPipeline

tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

print(tvm.__file__)

#TODO: support fp16
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")

# def clip_to_text_embeddings(pipe) -> tvm.IRModule:
#     # Define the wrapper torch.nn.Module for CLIP.
#     class CLIPModelWrapper(torch.nn.Module):
#         def __init__(self, clip):
#             super().__init__()
#             self.clip = clip

#         def forward(self, text_input_ids):
#             text_embeddings = self.clip(text_input_ids)[0]
#             return text_embeddings

#     clip = pipe.text_encoder_2
#     clip_to_text_embeddings = CLIPModelWrapper(clip)

#     # Create random input (77 is the maximum length).
#     text_input_ids = torch.rand((1, 77)).to(torch.int32)
#     # Capture CLIP's computational graph.
#     # mod = dynamo_capture_subgraphs(
#     #     clip_to_text_embeddings.forward,
#     #     text_input_ids,
#     #     keep_params_as_input=True,
#     # )
#     # assert len(mod.functions) == 1

#     # return tvm.IRModule({"clip": mod["subgraph_0"]})

#     graph = fx.symbolic_trace(clip_to_text_embeddings)
#     mod = from_fx(
#         graph,
#         [((1, 77), "int32")],
#         keep_params_as_input=True,
#     )
#     return tvm.IRModule({"clip2": mod["main"]})

# clip = clip_to_text_embeddings(pipe)
# print("successful import")

# print(pipe)

pipe.text_encoder_2.eval()

from web_stable_diffusion.utils import get_clip



tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

prompt = "a beautiful girl floating in galaxy"

text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

print("our result")
our_out = text_inputs.input_ids

for i in range(text_inputs.attention_mask.shape[1]):
    if text_inputs.attention_mask[0][i] == 0:
        our_out[0][i] = 0


# input = torch.rand((1, 77)).to(torch.int32)
text_input_ids = our_out.to(torch.int32)
print(text_input_ids)

with torch.no_grad():
    clip2 = get_clip(pipe)

    # text_input_ids = torch.rand((1, 77)).to(torch.int32)


    print("our out")
    out = clip2(text_input_ids)
    print(out)
    print("our pool")
    print(out.text_embeds.squeeze(1))
    print("our embd")
    print(out.hidden_states[-2])



    print("ref embd")
    ref_out = pipe.text_encoder_2(text_input_ids, output_hidden_states=True)
    print(ref_out.hidden_states[-2])

    print("ref pool")
    print(ref_out[0])

import numpy as np
np.testing.assert_allclose(out.text_embeds.squeeze(1).numpy(), ref_out[0].numpy(), atol=1e-6)
print("embd check success")
np.testing.assert_allclose(out.hidden_states[-2].numpy(), ref_out.hidden_states[-2].numpy())
print("pool check success")
