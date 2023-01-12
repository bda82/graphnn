import tensorflow as tf


def mixed_mode_support(scatter_fn):
    def _wrapper_mm_support(updates, indices, N):
        if len(updates.shape) == 3:
            updates = tf.transpose(updates, perm=(1, 0, 2))
        out = scatter_fn(updates, indices, N)
        if len(out.shape) == 3:
            out = tf.transpose(out, perm=(1, 0, 2))
        return out

    _wrapper_mm_support.__name__ = scatter_fn.__name__
    return _wrapper_mm_support
