import json

from box import Box
from fastapi import HTTPException, status


def load_meta_data(path: str) -> Box:
    """
    Load and return API meta-data configuration.

    This function reads a JSON configuration file
    from the provided path and wraps its content
    into a Box object to enable dot-notation access.

    Args:
        path (str): Path to the JSON configuration file.

    Returns:
        Box: Configuration object accessible via dot notation.

    Raises:
        Raises:
            HTTPException: If an error occurs during meta-data
                           configuration file loading, e.g.:
                 * If the file is not found at the given path.
                 * If the file cannot be accessed due to
                   insufficient permissions.
                 * If an I/O error occurs during reading operation.
                 * If the JSON content is malformed or invalid.
    """
    try:
        # Load meta data from JSON file
        with open(path, "r") as f:
            meta_data = json.load(f)

        # Wrap into Box for dot notation
        return Box(meta_data)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Meta-data configuration file not found at '{path}'",
        ) from e
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Permission denied while reading meta-data configuration file at '{path}'",
        ) from e
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"I/O error while reading the meta-data configuration file '{path}'",
        ) from e
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid JSON format in meta-data configuration file at '{path}'",
        ) from e
