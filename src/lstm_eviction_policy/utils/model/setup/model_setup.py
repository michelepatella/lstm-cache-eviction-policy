import torch
from utils.logs.log_utils import debug, info
from utils.model.LSTM import LSTM
from utils.model.utils.class_weights_calculator import calculate_class_weights
from utils.model.utils.optimizer_builder import build_optimizer


def model_setup(model_params, learning_rate, targets, config_settings):
    """
    Method to set up the training and testing processes.
    :param model_params: The model parameters.
    :param learning_rate: The learning rate.
    :param targets: The targets.
    :param config_settings: The configuration settings.
    :return: The device to use, the loss function, the model and the optimizer.
    """
    # initial message
    info("🔄 Model setup started...")

    # debugging
    debug(f"⚙️ Model params: {model_params}.")
    debug(f"⚙️ Learning rate: {learning_rate}.")
    debug(f"⚙️ Targets: {targets}.")

    try:
        # define the device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # get the class weights
        class_weights = calculate_class_weights(targets, config_settings)

        # debugging
        debug(f"⚙️ Class weights: {class_weights}.")

        # define the loss function
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights).float().to(device)
        )

        # define the LSTM model
        model = LSTM(model_params, config_settings).to(device)

        # define the optimizer
        optimizer = build_optimizer(model, learning_rate, config_settings)
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Model setup completed.")

    return (device, criterion, model, optimizer)
