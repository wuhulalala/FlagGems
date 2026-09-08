"""Normalize current Trident callable names for FlagGems include lookup."""


def configure_trident(enabled=True, skip=(), static=()):
    """Leave skipped wrappers in Gems and force validated static wrappers static."""
    import trident

    original = trident.jit

    def jit(fn=None, **kwargs):
        if not enabled:
            return fn if fn is not None else lambda f: f
        if fn is None:
            return lambda f: jit(f, **kwargs)
        if any(
            fn.__name__ == name or fn.__name__.startswith(name + "_wrapper_rank_")
            for name in skip
        ):
            return fn
        if any(
            fn.__name__ == name or fn.__name__.startswith(name + "_wrapper_rank_")
            for name in static
        ):
            kwargs["dynamic"] = False
        else:
            kwargs["dynamic"] = True
        return original(fn, **kwargs)

    trident.jit = jit


def prepare_registry():
    from trident.backend import TridentGraphModule

    import flag_gems

    for _, fn, *_ in flag_gems._FULL_CONFIG:
        if isinstance(fn, TridentGraphModule):
            fn.__name__ = fn.fn.__name__
    mapping = {}
    for entry in flag_gems._FULL_CONFIG:
        mapping.setdefault(entry[1].__name__, []).append(entry)
    flag_gems.FULL_CONFIG_BY_FUNC = mapping
    return flag_gems
