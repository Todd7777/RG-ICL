from pathlib import Path
import json
import csv
from .base import ClassificationDataset, ClassificationSample


class CheXpertDataset(ClassificationDataset):
    LABELS = [
        "no_finding",
        "enlarged_cardiomediastinum",
        "cardiomegaly",
        "lung_opacity",
        "lung_lesion",
        "edema",
        "consolidation",
        "pneumonia",
        "atelectasis",
        "pneumothorax",
        "pleural_effusion",
        "pleural_other",
        "fracture",
        "support_devices",
    ]

    @property
    def name(self) -> str:
        return "chexpert"

    @property
    def label_names(self) -> list:
        return self.LABELS

    @property
    def n_classes(self) -> int:
        return 14

    @property
    def is_multi_label(self) -> bool:
        return True

    def _parse_chexpert_label(self, value):
        if value == "" or value == "-1":
            return 0
        try:
            v = float(value)
            return 1 if v == 1.0 else 0
        except (ValueError, TypeError):
            return 0

    def _load(self):
        root = self.data_root / "chexpert"
        manifest_path = root / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            for entry in manifest["samples"]:
                if self.split == "all" or entry["split"] == self.split:
                    self.samples.append(ClassificationSample(
                        id=entry["id"],
                        image_path=str(root / entry["image_path"]),
                        split=entry["split"],
                        label=-1,
                        label_name="multi_label",
                        multi_label=entry["multi_label"],
                        metadata=entry.get("metadata", {}),
                    ))
            return

        for split_dir in ["reference", "test"]:
            split_path = root / split_dir
            if not split_path.exists():
                continue
            if self.split != "all" and split_dir != self.split:
                continue

            csv_path = split_path / "labels.csv"
            if not csv_path.exists():
                continue

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_rel = row.get("Path", row.get("path", ""))
                    img_path = split_path / img_rel
                    if not img_path.exists():
                        continue

                    multi_label = []
                    for label_name in self.LABELS:
                        col_name = label_name.replace("_", " ").title()
                        alt_col = label_name
                        value = row.get(col_name, row.get(alt_col, "0"))
                        multi_label.append(self._parse_chexpert_label(value))

                    sample_id = f"chexpert_{split_dir}_{Path(img_rel).stem}"
                    self.samples.append(ClassificationSample(
                        id=sample_id,
                        image_path=str(img_path),
                        split=split_dir,
                        label=-1,
                        label_name="multi_label",
                        multi_label=multi_label,
                    ))
