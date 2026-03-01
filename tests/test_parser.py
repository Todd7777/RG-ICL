import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inference.output_parser import OutputParser


def test_parse_classification_standard():
    parser = OutputParser()
    raw = "Label: glaucoma\nConfidence: 0.87"
    labels = ["non_glaucoma", "glaucoma"]

    result = parser.parse_classification(raw, "q1", labels)
    assert result.parse_success is True
    assert result.predicted_label == "glaucoma"
    assert result.predicted_label_idx == 1
    assert abs(result.confidence - 0.87) < 0.01
    print("PASS: test_parse_classification_standard")


def test_parse_classification_label_in_text():
    parser = OutputParser()
    raw = "Based on the fundus image, this appears to be a case of glaucoma with probability 0.92."
    labels = ["non_glaucoma", "glaucoma"]

    result = parser.parse_classification(raw, "q2", labels)
    assert result.parse_success is True
    assert result.predicted_label == "glaucoma"
    assert result.predicted_label_idx == 1
    assert result.confidence > 0.5
    print("PASS: test_parse_classification_label_in_text")


def test_parse_classification_multiclass():
    parser = OutputParser()
    raw = "Label: moderate_npdr\nConfidence: 0.75"
    labels = ["no_dr", "mild_npdr", "moderate_npdr", "severe_npdr", "proliferative_dr", "ungradable"]

    result = parser.parse_classification(raw, "q3", labels)
    assert result.parse_success is True
    assert result.predicted_label == "moderate_npdr"
    assert result.predicted_label_idx == 2
    assert abs(result.confidence - 0.75) < 0.01
    print("PASS: test_parse_classification_multiclass")


def test_parse_classification_empty():
    parser = OutputParser()
    result = parser.parse_classification("", "q4", ["a", "b"])
    assert result.parse_success is False
    assert result.predicted_label == ""
    print("PASS: test_parse_classification_empty")


def test_parse_multilabel():
    parser = OutputParser()
    raw = """cardiomegaly: present, 0.85
lung opacity: absent, 0.12
edema: present, 0.73
fracture: absent, 0.05"""

    labels = ["cardiomegaly", "lung_opacity", "edema", "fracture"]
    result = parser.parse_classification(raw, "q5", labels, is_multi_label=True)

    assert result.parse_success is True
    assert result.multi_label_predictions[0] == 1
    assert result.multi_label_predictions[1] == 0
    assert result.multi_label_predictions[2] == 1
    assert result.multi_label_predictions[3] == 0
    assert result.multi_label_confidences[0] > 0.5
    print("PASS: test_parse_multilabel")


def test_parse_multilabel_partial():
    parser = OutputParser()
    raw = "cardiomegaly: present\nedema: absent"
    labels = ["cardiomegaly", "lung_opacity", "edema", "fracture"]

    result = parser.parse_classification(raw, "q6", labels, is_multi_label=True)
    assert result.multi_label_predictions[0] == 1
    assert result.multi_label_predictions[2] == 0
    print("PASS: test_parse_multilabel_partial")


def test_parse_vqa_with_answer_prefix():
    parser = OutputParser()
    raw = "Answer: The chest X-ray shows bilateral pleural effusion."
    result = parser.parse_vqa(raw, "q7")
    assert result.parse_success is True
    assert "bilateral pleural effusion" in result.answer
    print("PASS: test_parse_vqa_with_answer_prefix")


def test_parse_vqa_plain_text():
    parser = OutputParser()
    raw = "The image shows no abnormalities."
    result = parser.parse_vqa(raw, "q8")
    assert result.parse_success is True
    assert result.answer == "The image shows no abnormalities."
    print("PASS: test_parse_vqa_plain_text")


def test_parse_vqa_empty():
    parser = OutputParser()
    result = parser.parse_vqa("", "q9")
    assert result.parse_success is False
    assert result.answer == ""
    print("PASS: test_parse_vqa_empty")


def test_parse_confidence_percentage():
    parser = OutputParser()
    raw = "Label: glaucoma\nConfidence: 87%"
    labels = ["non_glaucoma", "glaucoma"]

    result = parser.parse_classification(raw, "q10", labels)
    assert result.parse_success is True
    assert abs(result.confidence - 0.87) < 0.01
    print("PASS: test_parse_confidence_percentage")


def test_parse_classification_underscore_vs_space():
    parser = OutputParser()
    raw = "Label: mild npdr\nConfidence: 0.65"
    labels = ["no_dr", "mild_npdr", "moderate_npdr"]

    result = parser.parse_classification(raw, "q11", labels)
    assert result.parse_success is True
    assert result.predicted_label == "mild_npdr"
    assert result.predicted_label_idx == 1
    print("PASS: test_parse_classification_underscore_vs_space")


def test_parse_deterministic():
    parser = OutputParser()
    raw = "Label: glaucoma\nConfidence: 0.87"
    labels = ["non_glaucoma", "glaucoma"]

    r1 = parser.parse_classification(raw, "q12", labels)
    r2 = parser.parse_classification(raw, "q12", labels)
    assert r1.predicted_label == r2.predicted_label
    assert r1.predicted_label_idx == r2.predicted_label_idx
    assert r1.confidence == r2.confidence
    print("PASS: test_parse_deterministic")


if __name__ == "__main__":
    test_parse_classification_standard()
    test_parse_classification_label_in_text()
    test_parse_classification_multiclass()
    test_parse_classification_empty()
    test_parse_multilabel()
    test_parse_multilabel_partial()
    test_parse_vqa_with_answer_prefix()
    test_parse_vqa_plain_text()
    test_parse_vqa_empty()
    test_parse_confidence_percentage()
    test_parse_classification_underscore_vs_space()
    test_parse_deterministic()
    print("\nAll parser tests passed.")
