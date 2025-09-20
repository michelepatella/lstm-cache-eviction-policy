import torch
from torch.utils.data import Dataset
from utils.data.dataset.dataset_loader import load_dataset
from utils.logs.log_utils import debug, info


class AccessLogsDataset(Dataset):
    def _split_dataset(self, dataset_type, config_settings):
        """
        Method to split the dataset based on the dataset type requested.
        :param dataset_type: The dataset type requested ("training" or "testing").
        :param config_settings: The configuration settings.
        :return:
        """
        # initial message
        info("🔄 Dataset splitting started...")

        # debugging
        debug(f"⚙️ Splitting mode: {dataset_type}.")

        try:
            # define the splitting's index
            split_idx = int(len(self.data) * config_settings.training_perc)
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except ValueError as e:
            raise ValueError(f"ValueError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # debugging
        debug(f"⚙️ Split index: {split_idx}.")

        # split the dataset
        try:
            if dataset_type == "training":
                self.data = self.data[:split_idx]
            else:
                self.data = self.data[split_idx:]
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except IndexError as e:
            raise IndexError(f"IndexError: {e}.")
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # show a successful message
        info("🟢 Dataset split.")

    def _set_fields(self, data, config_settings):
        """
        Method to set the fields of the dataset.
        :param data: The data from which the fields are extracted.
        :param config_settings: The configuration settings.
        :return:
        """
        # initial message
        info("🔄 AccessLogsDataset fields setting started...")

        try:
            self.columns = data.columns.tolist()

            # assuming target is the last column
            self.features = self.columns[:-1]
            self.target = self.columns[-1]

            self.seq_len = config_settings.seq_len

            # debugging
            debug(f"⚙️ Dataset columns: {self.columns}.")
            debug(f"⚙️ Feature(s): {self.features}.")
            debug(f"⚙️ Target: {self.target}.")
            debug(f"⚙️ Sequence length: {self.seq_len}.")
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except IndexError as e:
            raise IndexError(f"IndexError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # show a successful message
        info("🟢 AccessLogsDataset fields set.")

    def __init__(self, dataset_type, config_settings):
        """
        Method to instantiate the access logs dataset.
        :param dataset_type: The type of dataset requested ("training" or "testing").
        :param config_settings: The configuration settings.
        """
        # initial message
        info("🔄 AccessLogsDataset initialization started...")

        # load the dataset
        df = load_dataset(config_settings)

        # debugging
        debug(f"⚙️ Dataset shape: {df.shape}.")

        try:
            # shift target column (requests)
            df[df.columns[-1]] = df[df.columns[-1]].astype(int) - 1
            # set data
            self.data = df.copy()
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except KeyError as e:
            raise KeyError(f"KeyError: {e}.")
        except ValueError as e:
            raise ValueError(f"ValueError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except IndexError as e:
            raise IndexError(f"IndexError: {e}.")
        except MemoryError as e:
            raise MemoryError(f"MemoryError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # split the dataset to assign data properly
        self._split_dataset(dataset_type, config_settings)

        # set the fields of the dataset
        self._set_fields(self.data, config_settings)

        # show a successful message
        info("🟢 AccessLogsDataset initialized.")

    def __len__(self):
        """
        Method to return the length of the access logs dataset.
        :return: The length of the access logs dataset.
        """
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        """
        Method to return features and the next value in the sequence.
        :param idx: The index of the access logs dataset.
        :return: The numerical features and the key.
        """
        # initial message
        info("🔄 AccessLogsDataset item retrieval started...")

        # debugging
        debug(f"⚙️ Getting item at index: {idx}.")

        try:
            # get the feature sequence of length seq_len
            seq_data = self.data.iloc[idx : idx + self.seq_len]

            # extract numerical features
            x_features = torch.tensor(
                seq_data[self.features].values.astype(float), dtype=torch.float
            )

            # extract key IDs
            x_keys = torch.tensor(
                seq_data[self.target].values.astype(int), dtype=torch.long
            )

            # extract target
            target_row = self.data.iloc[idx + self.seq_len]
            y_key = torch.tensor(int(target_row[self.target]), dtype=torch.long)

            # debugging
            debug(f"⚙️ Feature vector shape: {x_features.shape}.")
            debug(f"⚙️ Key shape: {x_keys.shape}")
            debug(f"⚙️ Target: {y_key.item()}.")
        except IndexError as e:
            raise IndexError(f"IndexError: {e}.")
        except KeyError as e:
            raise KeyError(f"KeyError: {e}.")
        except ValueError as e:
            raise ValueError(f"ValueError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # show a successful message
        info("🟢 AccessLogsDataset retrieved.")

        return (x_features, x_keys, y_key)

    @classmethod
    def from_dataframe(cls, df, config_settings):
        instance = cls.__new__(cls)
        instance.data = df.copy()
        instance._set_fields(df, config_settings)
        return instance
