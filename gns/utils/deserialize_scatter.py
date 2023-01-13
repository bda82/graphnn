from gns.scatter.op_dict import OP_DICT


def deserialize_scatter(scatter):
    """
    Scatter deserializer.

    Args:
        scatter: scatter
    
    Returns:
        deserialized scatter
    """
    if isinstance(scatter, str) and scatter in OP_DICT:
        return OP_DICT[scatter]
    elif callable(scatter):
        return scatter
    else:
        raise ValueError(
            f"scatter должен быть строкой или вызовом функции из следующего набора: {list(OP_DICT.keys())}"
        )
