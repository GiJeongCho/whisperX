import torch
import typing

def patch_torch_load():
    """Monkey patch torch.load to fix weights_only=True issue in PyTorch 2.6+"""
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    # Attempt to register safe globals
    try:
        from omegaconf.listconfig import ListConfig
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata, typing.Any])
    except ImportError:
        pass
    except Exception:
        pass
