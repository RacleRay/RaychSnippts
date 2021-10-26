import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, NamedTuple, Iterable
from raych.util import logger

def batchify(data, bsz, use_cuda):
    # code from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data


def nan_safe_tensor_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result


def cycle_iterator_function(iterator_function):
    """
    Functionally equivalent to `itertools.cycle(iterator_function())`, but this function does not
    cache the result of calling the iterator like `cycle` does.  Instead, we just call
    `iterator_function()` again whenever we get a `StopIteration`.  This should only be preferred
    over `itertools.cycle` in cases where you're sure you don't want the caching behavior that's
    done in `itertools.cycle`.
    """
    iterator = iter(iterator_function())
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterator_function())


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None and param.requires_grad:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def clamp_tensor(tensor, minimum, maximum):
    """
    Supports sparse and dense tensors.
    Returns a tensor with values clamped between the provided minimum and maximum,
    without modifying the original tensor.
    """
    if tensor.is_sparse:
        coalesced_tensor = tensor.coalesce()

        coalesced_tensor._values().clamp_(minimum, maximum)
        return coalesced_tensor
    else:
        return tensor.clamp(minimum, maximum)


def int_to_device(device: Union[int, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device < 0:
        return torch.device("cpu")
    return torch.device(device)


def move_to_device(obj, device: Union[torch.device, int]):
    """
    Given a structure (possibly) containing Tensors,
    move all the Tensors to the specified device (or do nothing, if they are already on
    the target device).
    """
    device = int_to_device(device)

    if isinstance(obj, torch.Tensor):
        # You may be wondering why we don't just always call `obj.to(device)` since that would
        # be a no-op anyway if `obj` is already on `device`. Well that works fine except
        # when PyTorch is not compiled with CUDA support, in which case even calling
        # `obj.to(torch.device("cpu"))` would result in an error.
        return obj if obj.device == device else obj.to(device=device)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = move_to_device(value, device)
        return obj
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = move_to_device(item, device)
        return obj
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj


def log_frozen_and_tunable_parameter_names(model: torch.nn.Module) -> None:
    frozen_parameter_names, tunable_parameter_names = get_frozen_and_tunable_parameter_names(model)

    logger.info("The following parameters are Frozen (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)

    logger.info("The following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)


def get_frozen_and_tunable_parameter_names(
    model: torch.nn.Module,
) -> Tuple[Iterable[str], Iterable[str]]:
    frozen_parameter_names = (
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    )
    tunable_parameter_names = (
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    return frozen_parameter_names, tunable_parameter_names
