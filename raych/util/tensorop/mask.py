import torch
from typing import List, Union, Tuple, Any


class ConfigurationError(Exception):
    """
    # Reference : allennlp

    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """
    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        """return tuple. This first element will be called to create the initial version of the object 
        when reconstruct this class object. The second element will be called to provide object`s member 
        values."""
        return type(self), (self.message,)

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message


def replace_masked_values(
    tensor: torch.Tensor, mask: torch.BoolTensor, replace_with: float
) -> torch.Tensor:
    """
    # Reference : allennlp

    Replaces all masked values in `tensor` with `replace_with`.  `mask` must be broadcastable
    to the same shape as `tensor`. We require that `tensor.dim() == mask.dim()`, as otherwise we
    won't know which dimensions of the mask to unsqueeze.

    This just does `tensor.masked_fill()`, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    `tensor.masked_fill(~mask, replace_with)`.
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError(
            "tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim())
        )
    return tensor.masked_fill(~mask, replace_with)


def masked_softmax(
    vector: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int = -1,
    memory_efficient: bool = False,
) -> torch.Tensor:
    """
    `torch.nn.functional.softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If `memory_efficient` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and `memory_efficient` is false, this function
    returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (
                result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
            )
        else:
            masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_log_softmax(vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1) -> torch.Tensor:
    """
    `torch.nn.functional.log_softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a log_softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular log_softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not `nan`.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you `nans`.

    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def masked_max(
    vector: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    To calculate max along certain dimensions on masked values

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate max, assume unmasked parts are already zeros
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate max
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the maximum values.
    """
    replaced_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


def masked_mean(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    """
    To calculate mean along certain dimensions on masked values

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


def masked_flip(padded_sequence: torch.Tensor, sequence_lengths: List[int]) -> torch.Tensor:
    """
    Flips a padded tensor along the time dimension without affecting masked entries.

    # Parameters

    padded_sequence : `torch.Tensor`
        The tensor to flip along the time dimension.
        Assumed to be of dimensions (batch size, num timesteps, ...)
    sequence_lengths : `torch.Tensor`
        A list containing the lengths of each unpadded sequence in the batch.

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of the same shape as padded_sequence.
    """
    assert padded_sequence.size(0) == len(
        sequence_lengths
    ), f"sequence_lengths length ${len(sequence_lengths)} does not match batch size ${padded_sequence.size(0)}"
    num_timesteps = padded_sequence.size(1)
    flipped_padded_sequence = torch.flip(padded_sequence, [1])
    sequences = [
        flipped_padded_sequence[i, num_timesteps - length :]
        for i, length in enumerate(sequence_lengths)
    ]
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min


def get_lengths_from_binary_sequence_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.

    # Parameters

    mask : `torch.BoolTensor`, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.

    # Returns

    `torch.LongTensor`
        A torch.LongTensor of shape (batch_size,) representing the lengths
        of the sequences in the batch.
    """
    return mask.sum(-1)


def get_mask_from_sequence_lengths(
    sequence_lengths: torch.Tensor, max_length: int
) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.

    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor