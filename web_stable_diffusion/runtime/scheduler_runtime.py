from typing import ClassVar, List, Type

import json
import numpy as np

import tvm
from tvm import relax


class Scheduler:
    scheduler_name: ClassVar[str]
    timesteps: List[tvm.nd.NDArray]

    def __init__(self, artifact_path: str, device) -> None:
        raise NotImplementedError()

    def step(
        self,
        vm: relax.VirtualMachine,
        model_output: tvm.nd.NDArray,
        sample: tvm.nd.NDArray,
        counter: int,
    ) -> tvm.nd.NDArray:
        raise NotImplementedError()


class PNDMScheduler(Scheduler):
    scheduler_name = "pndm"

    def __init__(self, artifact_path: str, device) -> None:
        with open(f"{artifact_path}/scheduler_pndm_consts.json", "r") as file:
            jsoncontent = file.read()
        scheduler_consts = json.loads(jsoncontent)

        def f_convert(data, dtype):
            return [tvm.nd.array(np.array(t, dtype=dtype), device) for t in data]

        self.timesteps = f_convert(scheduler_consts["timesteps"], "int32")
        self.sample_coeff = f_convert(scheduler_consts["sample_coeff"], "float32")
        self.alpha_diff = f_convert(scheduler_consts["alpha_diff"], "float32")
        self.model_output_denom_coeff = f_convert(
            scheduler_consts["model_output_denom_coeff"], "float32"
        )

        self.ets: List[tvm.nd.NDArray] = [
            tvm.nd.empty((1, 4, 64, 64), "float32", device)
        ] * 4
        self.cur_sample: tvm.nd.NDArray

    def step(
        self,
        vm: relax.VirtualMachine,
        model_output: tvm.nd.NDArray,
        sample: tvm.nd.NDArray,
        counter: int,
    ) -> tvm.nd.NDArray:
        if counter != 1:
            self.ets = self.ets[-3:]
            self.ets.append(model_output)

        if counter == 0:
            self.cur_sample = sample
        elif counter == 1:
            sample = self.cur_sample

        prev_latents = vm[f"pndm_scheduler_step_{min(counter, 4)}"](
            sample,
            model_output,
            self.sample_coeff[counter],
            self.alpha_diff[counter],
            self.model_output_denom_coeff[counter],
            self.ets[0],
            self.ets[1],
            self.ets[2],
            self.ets[3],
        )

        return prev_latents


class DPMSolverMultistepScheduler(Scheduler):
    scheduler_name = "multistep-dpm-solver"

    def __init__(self, artifact_path: str, device) -> None:
        with open(
            f"{artifact_path}/scheduler_dpm_solver_multistep_consts.json", "r"
        ) as file:
            jsoncontent = file.read()
        scheduler_consts = json.loads(jsoncontent)

        def f_convert(data, dtype):
            return [tvm.nd.array(np.array(t, dtype=dtype), device) for t in data]

        self.timesteps = f_convert(scheduler_consts["timesteps"], "int32")
        self.alpha = f_convert(scheduler_consts["alpha"], "float32")
        self.sigma = f_convert(scheduler_consts["sigma"], "float32")
        self.c0 = f_convert(scheduler_consts["c0"], "float32")
        self.c1 = f_convert(scheduler_consts["c1"], "float32")
        self.c2 = f_convert(scheduler_consts["c2"], "float32")

        self.last_model_output: tvm.nd.NDArray = tvm.nd.empty(
            (1, 4, 64, 64), "float32", device
        )

    def step(
        self,
        vm: relax.VirtualMachine,
        model_output: tvm.nd.NDArray,
        sample: tvm.nd.NDArray,
        counter: int,
    ) -> tvm.nd.NDArray:
        model_output = vm["dpm_solver_multistep_scheduler_convert_model_output"](
            sample, model_output, self.alpha[counter], self.sigma[counter]
        )
        prev_latents = vm["dpm_solver_multistep_scheduler_step"](
            sample,
            model_output,
            self.last_model_output,
            self.c0[counter],
            self.c1[counter],
            self.c2[counter],
        )
        self.last_model_output = model_output
        return prev_latents


class EulerDiscreteScheduler(Scheduler):
    scheduler_name = "euler-discrete-solver"

    def __init__(self, artifact_path: str, device) -> None:
        with open(
            f"{artifact_path}/scheduler_euler_discrete_consts.json", "r"
        ) as file:
            jsoncontent = file.read()
        scheduler_consts = json.loads(jsoncontent)

        def f_convert(data, dtype):
            return [tvm.nd.array(np.array(t, dtype=dtype), device) for t in data]

        self.timesteps = f_convert(scheduler_consts["timesteps"], "int32")
        self.sigma = f_convert(scheduler_consts["sigma"], "float32")

        # self.last_model_output: tvm.nd.NDArray = tvm.nd.empty(
        #     (1, 4, 64, 64), "float32", device
        # )

    def step(
        self,
        vm: relax.VirtualMachine,
        model_output: tvm.nd.NDArray,
        sample: tvm.nd.NDArray,
        counter: int,
    ) -> tvm.nd.NDArray:
        # model_output = vm["dpm_solver_multistep_scheduler_convert_model_output"](
        #     sample, model_output, self.alpha[counter], self.sigma[counter]
        # )
        prev_latents = vm["euler_discrete_scheduler_step"](
            sample,
            model_output,
            self.sigma[counter],
            self.sigma[counter+1]
        )
        # self.last_model_output = model_output
        return prev_latents
    
    def scale_model_input(self, vm, sample: tvm.nd.NDArray, counter: int) -> tvm.nd.NDArray:
        result = vm["euler_discrete_scheduler_scale"](sample, self.sigma[counter])
        return result

# def __init__(
#     self,
#     num_train_timesteps: int = 1000,
#     beta_start: float = 0.0001,
#     beta_end: float = 0.02,
#     beta_schedule: str = "linear",
#     trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
#     prediction_type: str = "epsilon",
#     interpolation_type: str = "linear",
#     use_karras_sigmas: Optional[bool] = False,
#     timestep_spacing: str = "linspace",
#     steps_offset: int = 0,
# ):
#     # if trained_betas is not None:
#     #     self.betas = torch.tensor(trained_betas, dtype=torch.float32)
#     # elif beta_schedule == "linear":
#     #     self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
#     if beta_schedule == "scaled_linear":
#         # this schedule is very specific to the latent diffusion model.
#         self.betas = (
#             torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
#         )
#     # elif beta_schedule == "squaredcos_cap_v2":
#     #     # Glide cosine schedule
#     #     self.betas = betas_for_alpha_bar(num_train_timesteps)
#     # else:
#     #     raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

#     self.alphas = 1.0 - self.betas
#     self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

#     sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
#     sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
#     self.sigmas = torch.from_numpy(sigmas)

#     # setable values
#     self.num_inference_steps = None
#     timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
#     self.timesteps = torch.from_numpy(timesteps)
#     self.is_scale_input_called = False
#     self.use_karras_sigmas = use_karras_sigmas


# def step(
#     self,
#     model_output: torch.FloatTensor,
#     timestep: Union[float, torch.FloatTensor],
#     sample: torch.FloatTensor,
#     s_churn: float = 0.0,
#     s_tmin: float = 0.0,
#     s_tmax: float = float("inf"),
#     s_noise: float = 1.0,
#     generator: Optional[torch.Generator] = None,
#     return_dict: bool = True,
# ) -> Union[EulerDiscreteSchedulerOutput, Tuple]:
#     """
#     Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
#     process from the learned model outputs (most often the predicted noise).

#     Args:
#         model_output (`torch.FloatTensor`): direct output from learned diffusion model.
#         timestep (`float`): current timestep in the diffusion chain.
#         sample (`torch.FloatTensor`):
#             current instance of sample being created by diffusion process.
#         s_churn (`float`)
#         s_tmin  (`float`)
#         s_tmax  (`float`)
#         s_noise (`float`)
#         generator (`torch.Generator`, optional): Random number generator.
#         return_dict (`bool`): option for returning tuple rather than EulerDiscreteSchedulerOutput class

#     Returns:
#         [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`] or `tuple`:
#         [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`] if `return_dict` is True, otherwise a
#         `tuple`. When returning a tuple, the first element is the sample tensor.

#     """


#     # if isinstance(timestep, torch.Tensor):
#     #     timestep = timestep.to(self.timesteps.device)

#     step_index = (self.timesteps == timestep).nonzero().item()
#     sigma = self.sigmas[step_index]

#     # gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
#     gamma = 0

#     noise = randn_tensor(
#         model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
#     )

#     eps = noise * s_noise
#     # sigma_hat = sigma * (gamma + 1)
#     sigma_hat = sigma

#     # if gamma > 0:
#     #     sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

#     # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
#     # NOTE: "original_sample" should not be an expected prediction_type but is left in for
#     # backwards compatibility
#     # if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
#     #     pred_original_sample = model_output
#     if self.config.prediction_type == "epsilon":
#         pred_original_sample = sample - sigma_hat * model_output
#     # elif self.config.prediction_type == "v_prediction":
#     #     # * c_out + input * c_skip
#     #     pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
#     # else:
#     #     raise ValueError(
#     #         f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
#     #     )

#     # 2. Convert to an ODE derivative
#     derivative = (sample - pred_original_sample) / sigma_hat

#     dt = self.sigmas[step_index + 1] - sigma_hat

#     prev_sample = sample + derivative * dt

#     if not return_dict:
#         return (prev_sample,)

#     return EulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


# latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]


########################################################################

schedulers: List[Type[Scheduler]] = [DPMSolverMultistepScheduler, PNDMScheduler]
