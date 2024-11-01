from .rpaddle_gd import MultNoisePaddle_GD  # SGD rpaddle
from .rpaddle_gd_class import (
    MultNoisePaddle_GD2_class,
)  # SGD rpaddle with theta depending on the class
from .mm_rapddle_id import MM_PADDLE_id  # Vanilla MM RPaddle
from .mm_rpaddle_reg import MM_RPADDLE_reg  # Regularised MM RPaddle
from .mm_rpaddle_class import (
    MM_PADDLE_class,
    MM_RPADDLE_Class_Reg,
)  # MM RPADDLE and regularised MMRPADDLE with thet depending on the class
from .mm_rpaddle_reg_diag_cov import (
    MM_RPADDLE_reg_sigmas,
    MM_RPADDLE_reg_diag,
)  # MM RPADDLE with some covariance estimate (homothety or diagonal)
from .em_rfsl import EM_RobustPaddle_ID  # EM RPaddle with identity covariance
from .em_rfsl_cov import (
    EM_RobustPaddle,
)  # EM RPaddle with covariance estimation (not used)
from .paddle import Paddle  # PADDLE
from .paddle_gd import Paddle_GD  # Gradient decent version of PADDLE
from .tim import TIM_GD, Alpha_TIM  # TIM and ALpha TIM (see corresponding papers)
from .rtim import RTIM_GD  # Attempt at making TIm robust
from .baseline import Baseline  # Simple gradient descent objective function
from .bdcspn import BDCSPN  # BDCSPN
from .ici import ICI  # ICI
from .laplacianshot import LaplacianShot  # LaplacianShot
from .pt_map import PT_MAP  # PT_MAP


def get_method_builder(backbone, device, args, log_file):
    # Initialize method classifier builder
    method_info = {
        "backbone": backbone,
        "device": device,
        "log_file": log_file,
        "args": args,
    }

    # few-shot methods
    if args.name_method == "RPADDLE":
        method_builder = MultNoisePaddle_GD(**method_info)
    elif args.name_method == "BASELINE":
        method_builder = Baseline(**method_info)
    elif args.name_method == "BDCSPN":
        method_builder = BDCSPN(**method_info)
    elif args.name_method == "ICI":
        method_builder = ICI(**method_info)
    elif args.name_method == "LAPLACIANSHOT":
        method_builder = LaplacianShot(**method_info)
    elif args.name_method == "PT_MAP":
        method_builder = PT_MAP(**method_info)
    elif args.name_method == "RPADDLE2_CLASS":
        method_builder = MultNoisePaddle_GD2_class(**method_info)
    elif args.name_method == "MM_RPADDLE_REG":
        method_builder = MM_RPADDLE_reg(**method_info)
    elif args.name_method == "MM_RPADDLE_ID":
        method_builder = MM_PADDLE_id(**method_info)
    elif args.name_method == "MM_RPADDLE_CLASS":
        method_builder = MM_PADDLE_class(**method_info)
    elif args.name_method == "MM_RPADDLE_CLASS_REG":
        method_builder = MM_RPADDLE_Class_Reg(**method_info)
    elif args.name_method == "MM_RPADDLE_REG_SIGMAS":
        method_builder = MM_RPADDLE_reg_sigmas(**method_info)
    elif args.name_method == "MM_RPADDLE_REG_DIAG":
        method_builder = MM_RPADDLE_reg_diag(**method_info)
    elif args.name_method == "PADDLE":
        method_builder = Paddle(**method_info)
    elif args.name_method == "PADDLE_GD":
        method_builder = Paddle_GD(**method_info)
    elif args.name_method == "TIM_GD":
        method_builder = TIM_GD(**method_info)
    elif args.name_method == "RTIM_GD":
        method_builder = RTIM_GD(**method_info)
    elif args.name_method == "ALPHA_TIM":
        method_builder = Alpha_TIM(**method_info)
    elif args.name_method == "EM-PADDLE-ID":
        method_builder = EM_RobustPaddle_ID(**method_info)
    elif args.name_method == "EM-PADDLE":
        method_builder = EM_RobustPaddle(**method_info)

    else:
        raise ValueError(
            "The method your entered does not exist or is not a few-shot method. Please check the spelling"
        )
    return method_builder
