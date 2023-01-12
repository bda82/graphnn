from dataclasses import dataclass


@dataclass
class DataTypes:
    as_numpy_dtype = "as_numpy_dtype"


def data_types_fabric():
    return DataTypes()
