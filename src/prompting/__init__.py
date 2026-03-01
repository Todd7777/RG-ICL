from .templates import ClassificationTemplate, VQATemplate
from .zero_shot import ZeroShotPrompter
from .naive_icl import NaiveICLPrompter
from .rg_icl import RGICLPrompter

PROMPTERS = {
    "zero_shot": ZeroShotPrompter,
    "naive_icl": NaiveICLPrompter,
    "rg_icl_global": RGICLPrompter,
    "rg_icl_global_spatial": RGICLPrompter,
}


def get_prompter(method: str, **kwargs):
    if method not in PROMPTERS:
        raise ValueError(f"Unknown method: {method}. Available: {list(PROMPTERS.keys())}")
    return PROMPTERS[method](**kwargs)
