import json
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassificationParsedOutput:
    query_id: str
    predicted_label: str = ""
    predicted_label_idx: int = -1
    confidence: float = 0.0
    multi_label_predictions: list = field(default_factory=list)
    multi_label_confidences: list = field(default_factory=list)
    raw_response: str = ""
    parse_success: bool = False

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "predicted_label": self.predicted_label,
            "predicted_label_idx": self.predicted_label_idx,
            "confidence": self.confidence,
            "multi_label_predictions": self.multi_label_predictions,
            "multi_label_confidences": self.multi_label_confidences,
            "parse_success": self.parse_success,
        }


@dataclass
class VQAParsedOutput:
    query_id: str
    answer: str = ""
    raw_response: str = ""
    parse_success: bool = False

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "answer": self.answer,
            "parse_success": self.parse_success,
        }


class OutputParser:
    def __init__(self):
        pass

    def _normalize_label(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r'[^a-z0-9_\s]', '', text)
        text = re.sub(r'\s+', '_', text)
        return text

    def _find_best_label_match(self, text: str, label_names: list) -> tuple:
        text_normalized = self._normalize_label(text)

        for idx, label in enumerate(label_names):
            if self._normalize_label(label) == text_normalized:
                return idx, label

        for idx, label in enumerate(label_names):
            if self._normalize_label(label) in text_normalized:
                return idx, label

        for idx, label in enumerate(label_names):
            if text_normalized in self._normalize_label(label):
                return idx, label

        return -1, ""

    def _extract_confidence(self, text: str) -> float:
        patterns = [
            r'[Cc]onfidence[:\s]+([0-9]*\.?[0-9]+)',
            r'[Pp]robability[:\s]+([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)\s*(?:%|percent)',
            r'\b(0\.\d+|1\.0|1\.00?)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                val = float(match.group(1))
                if val > 1.0:
                    val = val / 100.0
                return min(max(val, 0.0), 1.0)
        return 0.5

    def _parse_json_response(self, raw_response: str, label_names: list) -> tuple:
        """Try to extract label+confidence from a JSON response. Returns (idx, label, confidence) or (-1, '', 0.5)."""
        # Strip markdown code fences if present
        text = raw_response.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        # Find first JSON object in the response
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if not match:
            return -1, '', 0.5
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return -1, '', 0.5
        label_val = data.get('label', '')
        confidence_val = data.get('confidence', 0.5)
        idx, matched_label = self._find_best_label_match(str(label_val), label_names)
        try:
            conf = float(confidence_val)
            if conf > 1.0:
                conf = conf / 100.0
            conf = min(max(conf, 0.0), 1.0)
        except (ValueError, TypeError):
            conf = 0.5
        return idx, matched_label, conf

    def parse_classification(self, raw_response: str, query_id: str,
                              label_names: list, is_multi_label: bool = False) -> ClassificationParsedOutput:
        output = ClassificationParsedOutput(query_id=query_id, raw_response=raw_response)

        if not raw_response or not raw_response.strip():
            return output

        if is_multi_label:
            return self._parse_multi_label(raw_response, query_id, label_names)

        # 1. Try JSON parsing first (gpt-4o returns JSON reliably)
        idx, matched_label, conf = self._parse_json_response(raw_response, label_names)
        if idx >= 0:
            output.predicted_label = matched_label
            output.predicted_label_idx = idx
            output.confidence = conf
            output.parse_success = True
            return output

        # 2. Try structured "Label: <value>" pattern
        label_match = re.search(r'[Ll]abel[:\s]+(.+?)(?:\n|$)', raw_response)
        if label_match:
            label_text = label_match.group(1).strip()
            idx, matched_label = self._find_best_label_match(label_text, label_names)
            if idx >= 0:
                output.predicted_label = matched_label
                output.predicted_label_idx = idx
                output.confidence = self._extract_confidence(raw_response)
                output.parse_success = True
                return output

        # 3. Substring fallback — sort by length descending to avoid false partial matches
        #    e.g. avoid matching 'non glaucoma' inside 'no signs of glaucoma'
        ranked = sorted(enumerate(label_names), key=lambda x: len(x[1]), reverse=True)
        for idx, label in ranked:
            label_lower = label.lower().replace('_', ' ')
            # Require word-boundary match to avoid 'non glaucoma' matching inside 'glaucoma'
            if re.search(r'\b' + re.escape(label_lower) + r'\b', raw_response.lower()):
                output.predicted_label = label
                output.predicted_label_idx = idx
                output.confidence = self._extract_confidence(raw_response)
                output.parse_success = True
                return output

        output.confidence = self._extract_confidence(raw_response)
        return output

    def _parse_multi_label(self, raw_response: str, query_id: str,
                           label_names: list) -> ClassificationParsedOutput:
        output = ClassificationParsedOutput(
            query_id=query_id,
            raw_response=raw_response,
            multi_label_predictions=[0] * len(label_names),
            multi_label_confidences=[0.0] * len(label_names),
        )

        lines = raw_response.strip().split("\n")
        matched_any = False

        for line in lines:
            line_lower = line.lower().strip()
            for idx, label in enumerate(label_names):
                label_variants = [
                    label.lower(),
                    label.lower().replace("_", " "),
                ]
                for variant in label_variants:
                    if variant in line_lower:
                        present_patterns = [
                            r'\bpresent\b', r'\byes\b', r'\bpositive\b',
                            r'\bfound\b', r'\bdetected\b', r'\b1\b',
                        ]
                        absent_patterns = [
                            r'\babsent\b', r'\bno\b', r'\bnegative\b',
                            r'\bnot\s+found\b', r'\bnot\s+detected\b', r'\b0\b',
                        ]

                        is_present = any(re.search(p, line_lower) for p in present_patterns)
                        is_absent = any(re.search(p, line_lower) for p in absent_patterns)

                        if is_present and not is_absent:
                            output.multi_label_predictions[idx] = 1
                        elif is_absent:
                            output.multi_label_predictions[idx] = 0

                        conf_match = re.search(r'([0-9]*\.?[0-9]+)', line.split(",")[-1] if "," in line else line)
                        if conf_match:
                            val = float(conf_match.group(1))
                            if val > 1.0:
                                val = val / 100.0
                            output.multi_label_confidences[idx] = min(max(val, 0.0), 1.0)
                        else:
                            output.multi_label_confidences[idx] = 0.5

                        matched_any = True
                        break

        output.parse_success = matched_any
        return output

    def parse_vqa(self, raw_response: str, query_id: str) -> VQAParsedOutput:
        output = VQAParsedOutput(query_id=query_id, raw_response=raw_response)

        if not raw_response or not raw_response.strip():
            return output

        answer_match = re.search(r'[Aa]nswer[:\s]+(.+?)(?:\n\n|$)', raw_response, re.DOTALL)
        if answer_match:
            output.answer = answer_match.group(1).strip()
        else:
            output.answer = raw_response.strip()

        output.parse_success = bool(output.answer)
        return output
