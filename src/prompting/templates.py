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


CLASSIFICATION_SYSTEM_PROMPTS = {
    "lag": (
        "You are a medical imaging assistant specialized in radiological disease detection.\n"
        "You must make a binary diagnostic decision from a single medical image.\n"
        "You must reason only from the image content provided and not assume patient history.\n"
        "If visual evidence is insufficient, you must output low confidence rather than guess.\n"
        "Always separate visual evidence, reasoning, and final decision.\n"
        "You must output calibrated probabilities."
    ),
    "ddr": (
        "You are an ophthalmic image grading assistant.\n"
        "You must assign one of six diabetic retinopathy severity grades (0-5).\n"
        "You must identify microaneurysms, hemorrhages, exudates, neovascularization, and vessel abnormalities."
    ),
    "chexpert": (
        "You are a clinical chest X-ray interpretation assistant.\n"
        "You must detect 14 possible conditions.\n"
        "Each condition is independent."
    ),
    "breakhis": (
        "You are a histopathology classification assistant.\n"
        "You must classify microscopic tissue images into 8 tumor subtypes.\n"
        "You must use cellular morphology, gland formation, nuclear atypia, and tissue architecture."
    ),
}

CLASSIFICATION_ZERO_SHOT_PROMPT = (
    "Given the image, determine whether the target pathology is present.\n"
    "Return:\n"
    "• Binary label (0 = absent, 1 = present)\n"
    "• Probability that the pathology is present\n"
    "• One sentence describing the strongest visual evidence"
)

CLASSIFICATION_NAIVE_ICL_PROMPT = (
    "You will be shown several example cases with images and labels.\n"
    "Learn the visual patterns associated with positive and negative disease.\n"
    "Then classify the query image accordingly."
)

CLASSIFICATION_RG_ICL_PROMPT = (
    "You will be given:\n"
    "1. The query image\n"
    "2. A set of retrieved reference images most similar in anatomical structure and pathology.\n"
    "Use the reference images to guide your classification of the query image."
)

VQA_SYSTEM_PROMPT = (
    "You are a board-certified medical specialist with extensive experience interpreting medical images.\n"
    "Your task is to answer clinical visual questions accurately and conservatively based solely on "
    "the provided image and question."
)

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = (
    "You are a medical imaging assistant specialized in radiological disease detection.\n"
    "You must make a diagnostic decision from the provided medical image.\n"
    "You must reason only from the image content provided and not assume patient history.\n"
    "If visual evidence is insufficient, you must output low confidence rather than guess.\n"
    "Always separate visual evidence, reasoning, and final decision.\n"
    "You must output calibrated probabilities."
)


class ClassificationTemplate:
    @staticmethod
    def get_system_prompt(dataset_name: str = "", is_multi_label: bool = False) -> str:
        return CLASSIFICATION_SYSTEM_PROMPTS.get(dataset_name, DEFAULT_CLASSIFICATION_SYSTEM_PROMPT)

    @staticmethod
    def get_method_instruction(method: str = "zero_shot") -> str:
        if method == "naive_icl":
            return CLASSIFICATION_NAIVE_ICL_PROMPT
        elif method in ("rg_icl_global", "rg_icl_global_spatial"):
            return CLASSIFICATION_RG_ICL_PROMPT
        return CLASSIFICATION_ZERO_SHOT_PROMPT

    SYSTEM_PROMPT = DEFAULT_CLASSIFICATION_SYSTEM_PROMPT

    MULTI_LABEL_SYSTEM_PROMPT = CLASSIFICATION_SYSTEM_PROMPTS.get("chexpert", DEFAULT_CLASSIFICATION_SYSTEM_PROMPT)

    @staticmethod
    def format_query(image_path: str, label_names: list, is_multi_label: bool = False,
                     dataset_name: str = "", method: str = "zero_shot") -> list:
        instruction = ClassificationTemplate.get_method_instruction(method)
        if is_multi_label:
            labels_str = ", ".join(label_names)
            text = (
                f"{instruction}\n\n"
                f"For each of the following findings, indicate whether it is present or absent "
                f"and provide a confidence probability:\n{labels_str}\n\n"
                f"Format: <finding>: <present/absent>, <probability>"
            )
        else:
            labels_str = ", ".join(label_names)
            positive_class = label_names[-1]
            text = (
                f"{instruction}\n\n"
                f"Classify this image into one of: {labels_str}\n\n"
                f"You MUST respond using ONLY this exact JSON format, no other text:\n"
                f'{{"label": "<one of: {", ".join(label_names)}>", '
                f'"confidence": <probability that the image is {positive_class}, float 0.0-1.0>, '
                f'"evidence": "<one sentence>"}}'
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
    SYSTEM_PROMPT = VQA_SYSTEM_PROMPT

    @staticmethod
    def format_query(image_path: str, question: str) -> list:
        text = f"Question: {question}\nProvide your answer based on the medical image shown."
        return [_image_content(image_path), _text_content(text)]

    @staticmethod
    def format_reference(image_path: str, question: str, answer: str) -> list:
        text = f"Reference — Question: {question}\nAnswer: {answer}"
        return [_image_content(image_path), _text_content(text)]
