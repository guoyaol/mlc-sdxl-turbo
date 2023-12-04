import torch

from typing import Dict, List, Tuple

import tvm
from tvm import relax

from .models.unet_2d_condition import TVMUNet2DConditionModel
from .models.autoencoder_kl import AutoencoderKL
from .models.modeling_clip import CLIPTextModelWithProjection


def detect_available_torch_device() -> str:
    if tvm.metal().exist:
        return "mps"
    elif tvm.cuda().exist:
        return "cuda"
    raise ValueError("At least one GPU backend is expected to be enabled")


def get_unet(
    pipe,
    device_str: str,
    cross_attention_dim=2048,
    attention_head_dim=[5, 10, 20],
    use_linear_projection=True,
):
    model = TVMUNet2DConditionModel(
        act_fn = "silu", 
        addition_embed_type = "text_time", 
        addition_embed_type_num_heads = 64, 
        addition_time_embed_dim = 256, 
        attention_head_dim = [5,10,20], 
        block_out_channels = [320,640,1280], 
        center_input_sample = False, 
        class_embed_type = None, 
        class_embeddings_concat = False, 
        conv_in_kernel = 3, 
        conv_out_kernel = 3, 
        cross_attention_dim = 2048, 
        cross_attention_norm = None, 
        down_block_types = ["DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"], 
        downsample_padding = 1, 
        dual_cross_attention = False, 
        encoder_hid_dim = None, 
        encoder_hid_dim_type = None, 
        flip_sin_to_cos = True, 
        freq_shift = 0, 
        in_channels = 4, 
        layers_per_block = 2, 
        mid_block_only_cross_attention = None, 
        mid_block_scale_factor = 1, 
        mid_block_type = "UNetMidBlock2DCrossAttn", 
        norm_eps = 1e-05, 
        norm_num_groups = 32, 
        num_attention_heads = None, 
        num_class_embeds = None, 
        only_cross_attention = False, 
        out_channels = 4, 
        projection_class_embeddings_input_dim = 2816, 
        resnet_out_scale_factor = 1.0, 
        resnet_skip_time_act = False, 
        resnet_time_scale_shift = "default", 
        sample_size = 64, 
        time_cond_proj_dim = None, 
        time_embedding_act_fn = None, 
        time_embedding_dim = None, 
        time_embedding_type = "positional", 
        timestep_post_act = None, 
        transformer_layers_per_block = [1,2,10], 
        up_block_types = ["CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D"], 
        upcast_attention = False, 
        use_linear_projection = True
    )
    # pt_model_dict = pipe.unet.state_dict()
    # model_dict = {}
    # for name, tensor in pt_model_dict.items():
    #     # if name.endswith("ff.net.0.proj.weight") or name.endswith("ff.net.0.proj.bias"):
    #     #     w1, w2 = tensor.chunk(2, dim=0)
    #     #     model_dict[name.replace("proj", "proj1")] = w1
    #     #     model_dict[name.replace("proj", "proj2")] = w2
    #     #     continue
    #     # if (name.endswith("proj_in.weight") or name.endswith("proj_out.weight")) and len(tensor.shape) == 2:
    #     #     # Convert Linear weights to 1x1 conv2d weights. This is necessary for SD v2 which uses
    #     #     # use_linear_projection = True.
    #     #     model_dict[name] = torch.unsqueeze(torch.unsqueeze(tensor, -1), -1)
    #     #     continue
    #     model_dict[name] = tensor
    model.load_state_dict(pipe.unet.state_dict())
    return model


def merge_irmodules(*irmodules: tvm.IRModule) -> tvm.IRModule:
    merged_mod = tvm.IRModule()

    for mod in irmodules:
        for gv, func in mod.functions.items():
            merged_mod[gv] = func
    return merged_mod


def split_transform_deploy_mod(
    mod: tvm.IRModule, model_names: List[str], mod_deploy_entry_func: List[str]
) -> Tuple[tvm.IRModule, tvm.IRModule]:
    mod_transform = tvm.IRModule()
    mod_deploy = tvm.IRModule()

    transform_func_names = [name + "_transform_params" for name in model_names]
    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, tvm.tir.PrimFunc):
            mod_transform[gv] = func
            mod_deploy[gv] = func
        elif gv.name_hint in transform_func_names:
            mod_transform[gv] = func
        else:
            mod_deploy[gv] = func

    mod_transform = relax.transform.DeadCodeElimination(transform_func_names)(
        mod_transform
    )
    mod_deploy = relax.transform.DeadCodeElimination(mod_deploy_entry_func)(mod_deploy)

    return mod_transform, mod_deploy


def transform_params(
    mod_transform: tvm.IRModule, model_params: Dict[str, List[tvm.nd.NDArray]]
) -> Dict[str, List[tvm.nd.NDArray]]:
    ex = relax.build(mod_transform, target="llvm")
    vm = relax.vm.VirtualMachine(ex, tvm.cpu())
    new_params = dict()
    for name, params in model_params.items():
        new_params[name] = vm[name + "_transform_params"](params)
    return new_params


def save_params(params: Dict[str, List[tvm.nd.NDArray]], artifact_path: str) -> None:
    from tvm.contrib import tvmjs

    meta_data = {}
    param_dict = {}
    for model in ["unet", "vae", "clip", "clip2"]:
        meta_data[f"{model}ParamSize"] = len(params[model])
        for i, nd in enumerate(params[model]):
            param_dict[f"{model}_{i}"] = nd
    tvmjs.dump_ndarray_cache(param_dict, f"{artifact_path}/params", meta_data=meta_data)


def load_params(artifact_path: str, device) -> Dict[str, List[tvm.nd.NDArray]]:
    from tvm.contrib import tvmjs

    pdict = {}
    params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}/params", device)
    for model in ["vae", "unet", "clip", "clip2"]:
        plist = []
        size = meta[f"{model}ParamSize"]
        for i in range(size):
            plist.append(params[f"{model}_{i}"])
        pdict[model] = plist
    return pdict

def get_vae(
    pipe
):
    model = AutoencoderKL(
        act_fn = "silu",
        block_out_channels = [
            128,
            256,
            512,
            512
        ],
        down_block_types = [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D"
        ],
        in_channels = 3,
        latent_channels = 4,
        layers_per_block = 2,
        norm_num_groups = 32,
        out_channels = 3,
        sample_size = 1024,
        scaling_factor = 0.13025,
        up_block_types = [
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D"
        ]
    )
    # pt_model_dict = pipe.unet.state_dict()
    # model_dict = {}
    # for name, tensor in pt_model_dict.items():
    #     # if name.endswith("ff.net.0.proj.weight") or name.endswith("ff.net.0.proj.bias"):
    #     #     w1, w2 = tensor.chunk(2, dim=0)
    #     #     model_dict[name.replace("proj", "proj1")] = w1
    #     #     model_dict[name.replace("proj", "proj2")] = w2
    #     #     continue
    #     # if (name.endswith("proj_in.weight") or name.endswith("proj_out.weight")) and len(tensor.shape) == 2:
    #     #     # Convert Linear weights to 1x1 conv2d weights. This is necessary for SD v2 which uses
    #     #     # use_linear_projection = True.
    #     #     model_dict[name] = torch.unsqueeze(torch.unsqueeze(tensor, -1), -1)
    #     #     continue
    #     model_dict[name] = tensor
    model.load_state_dict(pipe.vae.state_dict())
    return model

def get_clip(pipe):
    class config():
        attention_dropout = 0.0
        bos_token_id = 0
        dropout = 0.0
        eos_token_id = 2
        hidden_act = "gelu"
        hidden_size = 1280
        initializer_factor = 1.0
        initializer_range = 0.02
        intermediate_size = 5120
        layer_norm_eps = 1e-05
        max_position_embeddings = 77
        model_type = "clip_text_model"
        num_attention_heads = 20
        num_hidden_layers = 32
        pad_token_id = 1
        projection_dim = 1280
        torch_dtype = "float16"
        transformers_version = "4.32.0.dev0"
        vocab_size = 49408
    
    config = config()
    model = CLIPTextModelWithProjection(config)
    model.load_state_dict(pipe.text_encoder_2.state_dict())
    return model

    
