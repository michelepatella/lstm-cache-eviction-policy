import torch
from torch.utils.data import DataLoader

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def extract_targets_from_data_loader(data_loader: DataLoader) -> torch.Tensor:
    """
    Extract all target tensors from a data loader.

    This function iterates through a data loader and extracts all target
    tensors. The collected targets are then concatenated into a single tensor.
    The function assumes the target tensor is the last element of the batch.

    Args:
        data_loader (DataLoader): DataLoader instance from which to extract
                                  targets.

    Returns:
        torch.Tensor: Concatenated tensor of all extracted targets.

    Raises:
        RuntimeError: If target extraction from data loader fails:
            * Accessing the target element fails because the batch is not a tuple/list
              of tensors or is empty (TypeError, IndexError).
            * Concatenating all extracted targets fails because the collected tensors
              have incompatible shapes or types (RuntimeError, TypeError).
    """
    try:
        debug(
            "Targets extraction from data loader started",
            extra={
                "batches_num": len(data_loader),
                "context": "Targets extraction from data loader",
            },
        )

        # Extract and collect all targets
        # from data loader
        all_targets = []
        for batch in data_loader:
            # Assumption: batch is a tuple/list of
            # tensors, with the target as the last element
            target = batch[-1]
            all_targets.append(target)

        # Concatenate extracted targets
        # as a unique tensor
        concatenated_targets = torch.cat(all_targets)

        debug(
            "Targets extraction from data loader completed",
            extra={
                "targets_extracted_num": concatenated_targets.size(0),
                "targets_shape": list(concatenated_targets.shape),
                "context": "Targets extraction from data loader",
            },
        )

        return concatenated_targets
    except (TypeError, IndexError, RuntimeError) as e:
        msg = "Targets extraction from data loader failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "batches_num": len(data_loader),
                "context": "Targets extraction from data loader",
            },
        )
        raise RuntimeError(msg) from e
