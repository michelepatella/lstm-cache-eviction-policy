import torch
from utils.logs.log_utils import debug, info


def build_optimizer(model, learning_rate, config_settings):
    """
    Method to build the optimizer.
    :param model: Model for which the optimizer will be built.
    :param learning_rate: Learning rate.
    :param config_settings: The configuration settings.
    :return: The created optimizer.
    """
    # initial message
    info("🔄 Optimizer building started...")

    # debugging
    debug(f"⚙️ Learning rate: {learning_rate}.")
    debug(f"⚙️ Optimizer type: {config_settings.optimizer_type}.")

    try:
        # define the optimizer
        if config_settings.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
            )
        elif config_settings.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=config_settings.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=config_settings.momentum,
            )
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except UnboundLocalError as e:
        raise UnboundLocalError(f"UnboundLocalError: {e}.")
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except AssertionError as e:
        raise AssertionError(f"AssertionError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Optimizer building completed.")

    return optimizer
