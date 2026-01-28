"""
Compatibility shim: expose the existing `kblam` package under the `lkif` name.
This allows a gradual migration: code can import `lkif.*` while the original
`kblam` package files remain in place.
"""

import importlib
import sys
import types

_this = sys.modules[__name__]

# Provide lazy proxy modules so importing `lkif.<mod>` works even if the
# original `kblam.<mod>` cannot be fully imported at shim-load time (e.g. missing
# heavy dependencies). The proxy will import the real module on first attribute access.
_submodules = ["models", "snippet_selector", "utils"]
_topmods = ["gpt_session", "kb_encoder"]

def _make_proxy(module_name: str, target_pkg: str = "kblam"):
    proxy = types.ModuleType(f"lkif.{module_name}")

    def __getattr__(name):
        real = importlib.import_module(f"{target_pkg}.{module_name}")
        # populate proxy namespace with attributes from real module
        for k, v in vars(real).items():
            if k not in ("__loader__", "__spec__", "__package__"):
                setattr(proxy, k, v)
        return getattr(real, name)

    proxy.__getattr__ = __getattr__
    return proxy

for sub in _submodules:
    proxy = _make_proxy(sub)
    sys.modules[f"lkif.{sub}"] = proxy
    setattr(_this, sub, proxy)

for m in _topmods:
    proxy = _make_proxy(m)
    sys.modules[f"lkif.{m}"] = proxy
    setattr(_this, m, proxy)

__all__ = ["models", "snippet_selector", "utils", "gpt_session", "kb_encoder"]
