from gns.scatter.op_dict import OP_DICT


def serialize_scatter(identifier):
    """
    Serialize scatter.
    
    Args:
        identifier: identifier
        
    Returns:
        serizlized scatter
    """
    if identifier in OP_DICT:
        return identifier
    elif hasattr(identifier, "__name__"):
        for k, v in OP_DICT.items():
            if v.__name__ == identifier.__name__:
                return k
        return None
