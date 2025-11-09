"""epochs_trainer.py

Module for training a model over multiple epochs with validation,
early stopping, and support for PyTorch Distributed Data Parallel (DDP).

This module provides the `train_epochs` function, which manages the
full training loop for a PyTorch model over a specified number of epochs.
It uses torch.multiprocessing.spawn to launch worker processes for DDP,
allowing for scalable training. Each worker trains the model on a subset of
the data (via DistributedSampler), while the master process handles
validation, early stopping checks, and tracking the best model weights.

Functions:
    train_epochs(
        num_epochs: int,
        model: torch.nn.Module,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        optimizer: Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        current_phase: str,
        config: Any,
    ) -> tuple[float, torch.nn.Module]
        Spawns worker processes for distributed training, collects the results
        from the master process, and returns the best model and loss.
    _train_epochs_worker(
        rank: int,
        num_workers: int,
        num_epochs: int,
        current_phase: str,
        model: torch.nn.Module,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        optimizer: Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        config: Config,
        return_queue: Queue,
    ) -> None
        The worker function executed by each DDP process, managing the
        training, validation, and early stopping logic for its assigned rank.
"""

import copy
from multiprocessing import Queue
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from components.const import (
    TRAINING_EPOCHS_DESC,
    TRAINING_WORKERS_JOIN,
    TRAINING_BACKEND_NCCL,
    TRAINING_BACKEND_GLOO,
    TRAINING_INIT_METHOD,
    TRAINING_MASTER_PROCESS_RANK,
)
from components.data_loader.builder import build_data_loader
from components.evaluation.model.evaluator import evaluate_model
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.model.best.checks_updates.checker_updater import (
    check_update_best_model,
)
from components.training.callbacks.early_stopping import EarlyStopping
from components.training.core.single_epoch_trainer import train_single_epoch
from const import (
    LOGS_PHASE_VALIDATION,
    HW_DEVICE_CUDA_NAME,
    HW_DEVICE_MPS_NAME,
)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from pipeline.config.pydantic.config import Config


def _train_epochs_worker(
    rank: int,
    num_workers: int,
    num_epochs: int,
    current_phase: str,
    model: torch.nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    config: Config,
    return_queue: Queue,
) -> None:
    """Worker function executed by each process during Distributed
    Data Parallel (DDP) training.

    This function initializes the distributed process group, sets up the
    model with DDP, wraps the training DataLoader with a DistributedSampler,
    and runs the main training and validation loop across all epochs.
    The master process is responsible for:
    1. Evaluating the model after each epoch.
    2. Checking and updating the best model weights.
    3. Applying the Early Stopping mechanism.
    4. Sending the final best model and average loss back to the
       main thread via a Queue.

    Args:
        rank (int): The current process rank.
        num_workers (int): The total number of processes/workers.
        num_epochs (int): The maximum number of epochs to run.
        current_phase (str): The current pipeline phase.
        model (torch.nn.Module): The model to be trained (pre-DDP wrapper).
        training_loader (DataLoader): The DataLoader containing the full
                                      training dataset.
        validation_loader (DataLoader): The DataLoader for validation.
        optimizer (Optimizer): The optimizer instance.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device for this worker.
        config (Config): The configuration object.
        return_queue (Queue): A multiprocessing Queue to pass results back
                              to the main thread.

    Returns:
        None

    Raises:
        RuntimeError: If epochs training fails:
            * Distributed process group initialization fails (RuntimeError).
            * Training or validation fails due to incompatible data types
              or shapes (TypeError).
            * Early stopping or best model tracking fails due to comparison
              errors or invalid state dictionary (TypeError).
    """
    try:
        info(
            "Epochs training started",
            extra={
                "epochs_num": num_epochs,
                "training_loader_len": len(training_loader),
                "validation_loader_len": len(validation_loader),
                "optimizer": str(optimizer),
                "criterion": str(criterion),
                "device": str(device),
                "context": "Epochs training",
            },
        )

        # Prepare configuration
        device_type = config.training.device.type
        training_shuffle = config.training.general.shuffle
        training_batch_size = config.training.general.batch_size

        # Configuration for distributing training
        dist.init_process_group(
            backend=TRAINING_BACKEND_NCCL
            if device_type == HW_DEVICE_CUDA_NAME
            else TRAINING_BACKEND_GLOO,
            init_method=TRAINING_INIT_METHOD,
            world_size=num_workers,
            rank=rank,
        )

        # Setup model with DDP
        model = DDP(
            model,
            device_ids=[rank]
            if device_type in (HW_DEVICE_CUDA_NAME, HW_DEVICE_MPS_NAME)
            else None,
        )

        # Initialize both training sampler
        # and loader
        training_sampler = DistributedSampler(
            training_loader.dataset,
            num_replicas=num_workers,
            rank=rank,
            shuffle=training_shuffle,
        )
        training_loader = build_data_loader(
            training_loader.dataset,
            training_batch_size,
            sampler=training_sampler,
        )

        # Prepare configuration
        es_patience = (
            config.validation.early_stopping.patience
            if current_phase == LOGS_PHASE_VALIDATION
            else config.training.early_stopping.patience
        )
        es_delta = (
            config.validation.early_stopping.delta
            if current_phase == LOGS_PHASE_VALIDATION
            else config.training.early_stopping.delta
        )

        # Instantiate early stopping
        es = (
            EarlyStopping(es_patience, es_delta)
            if rank == TRAINING_MASTER_PROCESS_RANK
            else None
        )

        # Initialization
        best_model_weights = (
            copy.deepcopy(model.state_dict())
            if rank == TRAINING_MASTER_PROCESS_RANK
            else None
        )
        best_avg_loss = (
            np.inf if rank == TRAINING_MASTER_PROCESS_RANK else None
        )

        # Train the model over each epoch
        epoch = None
        for epoch in tqdm(range(1, num_epochs + 1), desc=TRAINING_EPOCHS_DESC):
            # Train one epoch
            train_single_epoch(
                model,
                training_loader,
                optimizer,
                criterion,
                device,
                epoch,
            )

            if rank == TRAINING_MASTER_PROCESS_RANK:
                # Evaluate the model to get the
                # average loss after the current epoch
                avg_loss, *_ = evaluate_model(
                    model,
                    validation_loader,
                    criterion,
                    device,
                )

                # Check for an update in average loss
                best_avg_loss, new_model_weights = check_update_best_model(
                    avg_loss,
                    best_avg_loss,
                    model,
                )

                # Update best model weights (if any)
                if new_model_weights:
                    best_model_weights = new_model_weights

                # Early stopping check
                # (check whether to stop training process
                # as the number of epochs without improvement
                # in average loss exceeds the patience)
                es(avg_loss)
                if es.early_stop:
                    break

        # Apply the best weights
        # before returning the trained model
        model.load_state_dict(best_model_weights)

        info(
            "Epochs training completed",
            extra={
                "loss_avg_best": None
                if np.isinf(best_avg_loss) or np.isnan(best_avg_loss)
                else float(best_avg_loss),
                "epochs_run_num": epoch,
                "early_stop_triggered": es.early_stop,
                "context": "Epochs training",
            },
        )

        # Before terminating, destroy the
        # current process
        dist.destroy_process_group()

        # Keep track of results if and only if
        # the current process is the master
        if rank == TRAINING_MASTER_PROCESS_RANK:
            return_queue.put((best_avg_loss, model))
    except (RuntimeError, TypeError) as e:
        msg = "Epochs training failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "epochs_num": num_epochs,
                "training_loader_len": len(training_loader),
                "validation_loader_len": len(validation_loader),
                "optimizer": str(optimizer),
                "criterion": str(criterion),
                "device": str(device),
                "context": "Epochs training",
            },
        )
        raise RuntimeError(msg) from e


def train_epochs(
    num_epochs: int,
    model: torch.nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    current_phase: str,
    config: Any,
) -> tuple[float, torch.nn.Module]:
    """Manages and executes the distributed training loop across multiple
     epochs.

    This function sets up the required parameters and uses torch.multiprocessing.spawn
    to launch multiple worker processes, each running the `_train_epochs_worker`
    function in parallel for DDP training. It then waits for the master process to
    return the final results via a multiprocessing Queue.

    Args:
        num_epochs (int): The maximum number of epochs to run.
        model (torch.nn.Module): The model instance to be trained.
        training_loader (DataLoader): The DataLoader for the training data.
        validation_loader (DataLoader): The DataLoader for the validation data.
        optimizer (Optimizer): The optimizer instance.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The base device used for training.
        current_phase (str): The current pipeline phase.
        config (Any): The configuration object.

    Returns:
        tuple[float, torch.nn.Module]:
            - best_avg_loss: The best average validation loss achieved during training.
            - best_model: The best model corresponding to the lowest loss.
    """
    # Configuration for distributed training
    num_workers = config.training.general.num_workers
    return_queue = mp.Queue()

    # Spawn different workers for training
    mp.spawn(
        _train_epochs_worker,
        args=(
            num_workers,
            num_epochs,
            current_phase,
            model,
            training_loader,
            validation_loader,
            optimizer,
            criterion,
            device,
            config,
            return_queue,
        ),
        nprocs=num_workers,
        join=TRAINING_WORKERS_JOIN,
    )

    # Retrieve final results
    best_avg_loss, best_model = return_queue.get()

    return best_avg_loss, best_model
