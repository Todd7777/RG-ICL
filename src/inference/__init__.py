from .output_parser import OutputParser, ClassificationParsedOutput, VQAParsedOutput


def MLLMClient(*args, **kwargs):
    from .mllm_client import MLLMClient as _MLLMClient
    return _MLLMClient(*args, **kwargs)
