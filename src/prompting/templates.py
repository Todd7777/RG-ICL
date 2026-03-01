from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptMessage:
    role: str
    content: list


def _image_content(image_path: str):
    return {"type": "image_url", "image_url": {"url": image_path}}


def _text_content(text: str):
    return {"type": "text", "text": text}


class ClassificationTemplate:
    SYSTEM_PROMPT = (
        "You are a medical imaging expert. You will be shown medical images and asked to "
        "provide a diagnosis. Respond with the diagnosis label and a confidence probability "
        "between 0 and 1. Format your response as:\n"
        "Label: <label>\nConfidence: <probability>"
    )

    MULTI_LABEL_SYSTEM_PROMPT = (
        "You are a medical imaging expert. You will be shown medical images and asked to "
        "identify all applicable findings. For each possible finding, respond with whether "
        "it is present and a confidence probability between 0 and 1. Format your response as:\n"
        "<finding>: <present/absent>, <probability>\n"
        "for each finding listed."
    )

    @staticmethod
    def format_query(image_path: str, label_names: list, is_multi_label: bool = False,
                     dataset_name: str = "") -> list:
        if is_multi_label:
            labels_str = ", ".join(label_names)
            text = (
                f"Examine this medical image from the {dataset_name} dataset. "
                f"For each of the following findings, indicate whether it is present or absent "
                f"and provide a confidence probability: {labels_str}"
            )
        else:
            labels_str = ", ".join(label_names)
            text = (
                f"Examine this medical image from the {dataset_name} dataset. "
                f"Classify it into one of the following categories: {labels_str}. "
                f"Provide your diagnosis label and confidence."
            )
        return [_image_content(image_path), _text_content(text)]

    @staticmethod
    def format_reference(image_path: str, label_name: str, is_multi_label: bool = False,
                         multi_label: list = None, label_names: list = None) -> list:
        if is_multi_label and multi_label is not None and label_names is not None:
            findings = []
            for name, val in zip(label_names, multi_label):
                status = "present" if val == 1 else "absent"
                findings.append(f"{name}: {status}")
            label_text = "; ".join(findings)
        else:
            label_text = label_name
        text = f"Reference image — Diagnosis: {label_text}"
        return [_image_content(image_path), _text_content(text)]


class VQATemplate:
    SYSTEM_PROMPT = (
        "You are a medical imaging expert. You will be shown a medical image along with a "
        "question about it. Provide a clear, accurate answer based on the visual evidence."
    )

    @staticmethod
    def format_query(image_path: str, question: str) -> list:
        text = f"Question: {question}\nProvide your answer based on the medical image shown."
        return [_image_content(image_path), _text_content(text)]

    @staticmethod
    def format_reference(image_path: str, question: str, answer: str) -> list:
        text = f"Reference — Question: {question}\nAnswer: {answer}"
        return [_image_content(image_path), _text_content(text)]
