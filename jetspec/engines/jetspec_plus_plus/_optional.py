"""Defer optional accelerator imports; execution still requires the real packages."""
import importlib
import importlib.util
from importlib import metadata
import os
import torch


class _Deferred:
    def __init__(self, module, attributes=()):
        self._module = module
        self._attributes = attributes

    def _load(self):
        value = importlib.import_module(self._module)
        for name in self._attributes:
            value = getattr(value, name)
        return value

    def __getattr__(self, name):
        return _Deferred(self._module, (*self._attributes, name))

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getitem__(self, grid):
        return self._load()[grid]


class _UnavailableKernel:
    def __init__(self, function):
        self.function = function

    def __getitem__(self, grid):
        raise ModuleNotFoundError('Install the jetspec-plus-plus-engine CUDA extra to launch kernels')

    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError('Install the jetspec-plus-plus-engine CUDA extra to launch kernels')


class _UnavailableTriton(_Deferred):
    def jit(self, function=None, **kwargs):
        return _UnavailableKernel(function) if function else _UnavailableKernel


class _Language(_Deferred):
    # An annotation marker only; no kernel can execute without Triton.
    constexpr = type('constexpr', (), {})


def optional(name):
    if importlib.util.find_spec(name.split('.')[0]) is not None:
        try:
            return importlib.import_module(name)
        except Exception:
            # Some environments have partially installed accelerator packages
            # (e.g., stale editable paths). Fall back to deferred imports so
            # non-accelerated backends can still run.
            return _Deferred(name)
    return _Deferred(name)


if getattr(torch.version, "hip", None):
    # ROCm wheels can ship Triton below AITER's strict gluon floor. The ROCm
    # backend docs explicitly allow system Triton via this env gate.
    os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

try:
    triton = importlib.import_module('triton')
    tl = importlib.import_module('triton.language')
except ModuleNotFoundError:
    triton = _UnavailableTriton('triton')
    tl = _Language('triton.language')

flashinfer = optional('flashinfer')
sampling = optional('flashinfer.sampling')
xxhash = optional('xxhash')
wandb = optional('wandb')


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return 'unavailable'
