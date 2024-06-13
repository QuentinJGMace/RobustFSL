from .rpaddle_gd import RobustPaddle_GD
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
    if args.name_method == "RPADDLE":
        method_builder = RobustPaddle_GD(**method_info)
    elif args.name_method == "PADDLE":
        method_builder = Paddle(**method_info)
    elif args.name_method == "PADDLE_GD":
        method_builder = Paddle_GD(**method_info)
    elif args.name_method == "TIM_GD":
        method_builder = TIM_GD(**method_info)
    elif args.name_method == "ALPHA_TIM":
        method_builder = Alpha_TIM(**method_info)

    else:
        raise ValueError(
            "The method your entered does not exist or is not a few-shot method. Please check the spelling"
        )
    return method_builder
