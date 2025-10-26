from components.logs.levels.error_logger import error


def extract_features_target_from_dataset_columns(
    columns: list[str],
) -> tuple[list[str], str]:
    """Extract features and target from dataset columns.

    This function extracts both features and target from provided dataset
    columns, assuming the last column is the target while all the other ones
    are features.

    Args:
        columns (list[str]): List of dataset columns to extract features
                             and target for.

    Returns:
        tuple[list[str], str]: List of features and target extracted.

    Raises:
        RuntimeError: If extracting features or target fails:
            * Columns list is empty (IndexError).
    """
    try:
        # Extract features and target
        features = columns[:-1]
        target = columns[-1]

        return features, target
    except IndexError as e:
        msg = "Features and target extraction from dataset columns failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "columns_provided": columns,
                "num_columns": len(columns) if columns else 0,
                "context": "Features and target extraction from dataset columns",
            },
        )
        raise RuntimeError(msg) from e
