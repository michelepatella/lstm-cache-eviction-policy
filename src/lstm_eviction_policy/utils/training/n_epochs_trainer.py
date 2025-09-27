import copy

from tqdm import tqdm
from training.utils.EarlyStopping import (
    EarlyStopping,
)
from utils.logs.log_utils import debug, info
from utils.model.evaluation.model_evaluator import (
    evaluate_model,
)
from utils.training.one_epoch_trainer import (
    train_one_epoch,
)


def train_n_epochs(
    epochs,
    model,
    training_loader,
    optimizer,
    criterion,
    device,
    config_settings,
    validation_loader=None,
):
    """
    Method to train the model a specified number of epochs.
    :param epochs: Number of epochs.
    :param model: The model to be trained.
    :param training_loader: The training loader.
    :param optimizer: The optimizer to be used.
    :param criterion: The loss function.
    :param device: The device to be used.
    :param config_settings: The configuration settings.
    :param early_stopping: Whether to apply early stopping or not.
    :param validation_loader: Validation data loader.
    :return: The average loss and the best trained model.
    """
    # initial message
    info("🔄 Train n-epochs started...")

    # debugging
    debug(f"⚙️ Number of epochs: {epochs}.")
    debug(f"⚙️ Training loader size: {len(training_loader)}.")
    debug(f"⚙️ Optimizer to use: {optimizer}.")
    debug(f"⚙️ Criterion to use: {criterion}.")
    debug(f"⚙️ Device to use: {device}.")
    debug(
        f"⚙️ Validation loader: {'Received' if validation_loader is not None else 'Not received'}."
    )

    try:
        # initialize data
        tot_loss = 0.0
        num_epochs_run = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float("inf")
        es = None

        # instantiate early stopping object
        es = EarlyStopping(config_settings)

        # n-epochs learning
        for _ in tqdm(range(1, epochs + 1), desc="Training"):
            # train the model
            train_one_epoch(
                model,
                training_loader,
                optimizer,
                criterion,
                device,
            )

            # increase number of epochs by one
            num_epochs_run += 1

            if validation_loader is not None:
                # get the validation average loss
                avg_loss, *_ = evaluate_model(
                    model,
                    validation_loader,
                    criterion,
                    device,
                    config_settings,
                )
                tot_loss = tot_loss + avg_loss

                # save the model weights if it is the new best one
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                # early stopping logic
                es(avg_loss)
                # check whether to stop
                if es.early_stop:
                    info("🛑 Early stopping triggered.")
                    info("🟢 Train n-epochs completed.")
                    break

        # show the best validation loss obtained
        info(f"🏆 Best validation loss achieved: {best_loss}")
        info(f"ℹ️ No. of epochs run: {num_epochs_run}")

        if validation_loader is not None:
            # load best weights to the model
            model.load_state_dict(best_model_wts)
    except NameError as e:
        raise NameError(f"NameError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except LookupError as e:
        raise LookupError(f"LookupError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # debugging
    debug(f"⚙️ Number of epochs run: {num_epochs_run}.")

    # show a successful message
    info("🟢 Train n-epochs completed.")

    # check if the avg loss needs to be returned
    if validation_loader:
        return best_loss, model
    else:
        return None, model
