from tvm import meta_schedule as ms
import tvm
from tvm import relax

target = tvm.target.Target(
            "webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm"
        )

from dist.before_scheduling import Module
mod_deploy = Module

with target, tvm.transform.PassContext(opt_level=3):
    mod_deploy = tvm.tir.transform.DefaultGPUSchedule()(mod_deploy)

ex = relax.build(mod=mod_deploy, target=target)
ex.export_library("dist/sdxl_turbo.wasm")