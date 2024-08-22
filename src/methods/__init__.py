from .rpaddle_gd import MutlNoisePaddle_GD
from .rpaddle_gd2 import MultNoisePaddle_GD2
from .rpaddle_gd_class import MultNoisePaddle_GD2_class
from .mm_rapddle_id import MM_PADDLE_id
from .mm_rpaddle_reg import MM_RPADDLE_reg
from .mm_rpaddle_class import MM_PADDLE_class
from .mm_rpaddle_reg_diag_cov import MM_RPADDLE_reg_sigmas
from .em_rfsl import EM_RobustPaddle_ID
from .em_rfsl_cov import EM_RobustPaddle
from .paddle import Paddle
from .paddle_gd import Paddle_GD
from .tim import TIM_GD, Alpha_TIM
from .rtim import RTIM_GD


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
        method_builder = MutlNoisePaddle_GD(**method_info)
    elif args.name_method == "RPADDLE2":
        method_builder = MultNoisePaddle_GD2(**method_info)
    elif args.name_method == "RPADDLE2_CLASS":
        method_builder = MultNoisePaddle_GD2_class(**method_info)
    elif args.name_method == "MM_RPADDLE_REG":
        method_builder = MM_RPADDLE_reg(**method_info)
    elif args.name_method == "MM_RPADDLE_ID":
        method_builder = MM_PADDLE_id(**method_info)
    elif args.name_method == "MM_RPADDLE_CLASS":
        method_builder = MM_PADDLE_class(**method_info)
    elif args.name_method == "MM_RPADDLE_REG_SIGMAS":
        method_builder = MM_RPADDLE_reg_sigmas(**method_info)
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
