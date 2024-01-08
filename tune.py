import tvm
import pickle

mod = pickle.load(open("dist/before_scheduling.pkl", "rb"))

def tune(mod: tvm.IRModule) -> None:
    from tvm import meta_schedule as ms

    ms.relax_integration.tune_relax(
        mod=mod,
        target=tvm.target.Target("apple/m1-gpu-restricted"),
        params={},
        builder=ms.builder.LocalBuilder(
            max_workers=5,
        ),
        runner=ms.runner.LocalRunner(),
        work_dir="log_db",
        max_trials_global=500,
        max_trials_per_task=500,
        strategy=ms.search_strategy.EvolutionarySearch(init_min_unmeasured=10, max_fail_count=20),
    )

tune(mod)