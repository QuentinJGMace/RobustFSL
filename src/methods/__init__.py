from .rpaddle_gd import MutlNoisePaddle_GD, MutlNoisePaddle_GD_id
from .rpaddle_gd2 import MutlNoisePaddle_GD2, MutlNoisePaddle_GD_id2
from .mm_rapddle_id import MM_PADDLE_id, MM_PADDLE_glasso
from .em_rfsl import EM_RobustPaddle_ID
from .em_rfsl_cov import EM_RobustPaddle
from .paddle import Paddle
from .paddle_gd import Paddle_GD
from .tim import TIM_GD, Alpha_TIM


def get_method_builder(backbone, device, args, log_file):
    # Initialize method classifier builder
    method_info = {
        "backbone": backbone,
        "device": device,
        "log_file": log_file,
        "args": args,
    }

    # few-shot methods
    if args.name_method == "RPADDLE" and args.id_cov:
        method_builder = MutlNoisePaddle_GD_id(**method_info)
    elif args.name_method == "RPADDLE2" and args.id_cov:
        method_builder = MutlNoisePaddle_GD_id2(**method_info)
    elif args.name_method == "RPADDLE2_COV":
        method_builder = MutlNoisePaddle_GD2(**method_info)
    elif args.name_method == "MM_RPADDLE_ID":
        method_builder = MM_PADDLE_id(**method_info)
    elif args.name_method == "MM_RPADDLE_GLASSO":
        method_builder = MM_PADDLE_glasso(**method_info)
    elif args.name_method == "RPADDLE" and not args.id_cov:
        method_builder = MutlNoisePaddle_GD(**method_info)
    elif args.name_method == "PADDLE":
        method_builder = Paddle(**method_info)
    elif args.name_method == "PADDLE_GD":
        method_builder = Paddle_GD(**method_info)
    elif args.name_method == "TIM_GD":
        method_builder = TIM_GD(**method_info)
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
