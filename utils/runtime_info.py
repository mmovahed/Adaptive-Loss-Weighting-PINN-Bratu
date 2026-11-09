import os
import platform
import sys

import numpy as np
import torch

try:
    import scipy
except Exception:
    scipy = None

from configs import bratu_config as cfg




def _cpu_model():
    try:
        with open("/proc/cpuinfo", "r") as handle:
            for line in handle:
                if line.lower().startswith("model name") and ":" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _env(name, default="<unset>"):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def print_runtime_config(method, outer_seed, candidate_seeds, device, extra=None):
    print("\n" + "=" * 78)
    print("CLUSTER / RUNTIME CONFIGURATION")
    print("=" * 78)
    print("Protocol version       : {}".format(cfg.PROTOCOL_VERSION))
    print("Method                 : {}".format(method))
    print("Host                   : {}".format(platform.node()))
    print("Platform               : {}".format(platform.platform()))
    print("Machine                : {}".format(platform.machine()))
    print("CPU model              : {}".format(_cpu_model()))
    print("libc                   : {}".format(" ".join(x for x in platform.libc_ver() if x) or "unknown"))
    print("PBS job id             : {}".format(_env("PBS_JOBID")))
    print("PBS array id           : {}".format(_env("PBS_ARRAYID", _env("PBS_ARRAY_INDEX"))))
    print("PBS queue              : {}".format(_env("PBS_QUEUE", _env("PBS_O_QUEUE", "lumer"))))
    print("Python executable      : {}".format(sys.executable))
    print("Python version         : {}".format(sys.version.replace("\n", " ")))
    print("NumPy version          : {}".format(np.__version__))
    print("SciPy version          : {}".format("unavailable" if scipy is None else scipy.__version__))
    print("PyTorch version        : {}".format(torch.__version__))
    print("PyTorch CUDA build     : {}".format(getattr(torch.version, "cuda", None)))
    print("CUDA available         : {}".format(torch.cuda.is_available()))
    print("Selected device        : {}".format(device))
    print("Torch CPU threads      : {}".format(torch.get_num_threads()))
    print("OMP_NUM_THREADS        : {}".format(_env("OMP_NUM_THREADS")))
    print("MKL_NUM_THREADS        : {}".format(_env("MKL_NUM_THREADS")))
    print("PYTHONHOME             : {}".format(_env("PYTHONHOME")))
    print("PYTHONPATH             : {}".format(_env("PYTHONPATH")))
    print("LD_LIBRARY_PATH        : {}".format(_env("LD_LIBRARY_PATH")))
    print("-" * 78)
    print("Lambda                 : {}".format(cfg.LAMBDA))
    print("Outer seed             : {}".format(outer_seed))
    print("Candidate count        : {}".format(cfg.SELECTION_CANDIDATES))
    print("Candidate seed stride  : {}".format(cfg.CANDIDATE_SEED_STRIDE))
    print("Candidate seeds        : {}".format(", ".join(str(x) for x in candidate_seeds)))
    print("PINN architecture      : 1-{}-1".format("-".join(str(x) for x in cfg.HIDDEN_LAYERS)))
    print("Learning rate          : {}".format(cfg.LEARNING_RATE))
    print("Interior collocation   : {}".format(cfg.N_COLLOCATION))
    print("Interior validation    : {}".format(cfg.N_VALIDATION))
    print("Test points            : {}".format(cfg.N_TEST))
    print("Boundary loss          : mean squared error over x=0,1")
    print("PINN updates/candidate : {}".format(cfg.TOTAL_PINN_STEPS))
    print("DDQN train episodes    : {}".format(cfg.DDQN_TRAIN_EPISODES))
    print("Warm-up steps          : {}".format(cfg.WARMUP_STEPS))
    print("Decision interval      : {}".format(cfg.DECISION_INTERVAL))
    print("Selection protocol     : {}".format(cfg.SELECTION_PROTOCOL))
    print("Selection criterion    : unweighted Lf_val + Lb_val")
    print("Reference in selection : NO")
    print("Post-selection audit   : test metrics saved for ALL candidates")
    print("LRA smoothing          : {}".format(cfg.LRA_SMOOTHING))
    print("LRA update interval    : {}".format(cfg.LRA_UPDATE_INTERVAL))
    print("ReLoBRaLo alpha        : {}".format(cfg.RELOBRALO_ALPHA))
    print("ReLoBRaLo rho P        : {}".format(cfg.RELOBRALO_RHO))
    print("ReLoBRaLo temperature  : {}".format(cfg.RELOBRALO_TEMPERATURE))
    if extra:
        for key in sorted(extra.keys()):
            print("{:<23}: {}".format(str(key), extra[key]))
    print("=" * 78)
    sys.stdout.flush()
