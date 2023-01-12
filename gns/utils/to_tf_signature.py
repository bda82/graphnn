def to_tf_signature(signature):
    """
    Converts a dataset signature to a TensorFlow signature.

    Args:
        signature: signature of the dataset.
    
    Returns:
        TensorFlow signature.
    """
    output = []
    keys = ["x", "a", "e", "i"]

    for k in keys:
        if k in signature:
            shape = signature[k]["shape"]
            dtype = signature[k]["dtype"]
            spec = signature[k]["spec"]
            output.append(spec(shape, dtype))
    
    output = tuple(output)
    
    if "y" in signature:
        shape = signature["y"]["shape"]
        dtype = signature["y"]["dtype"]
        spec = signature["y"]["spec"]
        output = (output, spec(shape, dtype))

    return output
