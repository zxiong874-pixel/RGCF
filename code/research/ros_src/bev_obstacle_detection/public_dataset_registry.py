"""Helpers for public dataset onboarding manifests."""

from pathlib import Path

import yaml


REQUIRED_DATASET_FIELDS = ("id", "name", "homepage", "local_root", "modalities", "target_tasks")


def load_public_dataset_registry(path):
    registry_path = Path(path).expanduser().resolve()
    with registry_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    datasets = data.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("public dataset registry must contain a list named datasets")
    for item in datasets:
        missing = [field for field in REQUIRED_DATASET_FIELDS if field not in item]
        if missing:
            raise ValueError("dataset entry missing required fields: {}".format(", ".join(missing)))
    return data


def dataset_by_id(registry, dataset_id):
    for dataset in registry.get("datasets", []):
        if dataset.get("id") == dataset_id:
            return dataset
    raise KeyError("unknown public dataset id: {}".format(dataset_id))


def create_dataset_skeleton(dataset):
    root = Path(dataset["local_root"]).expanduser()
    paths = [
        root,
        root / "raw",
        root / "converted",
        root / "rosbags",
        root / "metadata",
        root / "README.md",
    ]
    for path in paths:
        if path.suffix:
            if not path.exists():
                path.write_text(_dataset_readme(dataset), encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)
    return root


def _dataset_readme(dataset):
    lines = [
        "# {}".format(dataset.get("name", dataset.get("id", "public dataset"))),
        "",
        "- id: `{}`".format(dataset.get("id", "")),
        "- role: `{}`".format(dataset.get("role", "")),
        "- homepage: {}".format(dataset.get("homepage", "")),
    ]
    if dataset.get("devkit"):
        lines.append("- devkit: {}".format(dataset["devkit"]))
    if dataset.get("paper"):
        lines.append("- paper: {}".format(dataset["paper"]))
    lines.extend(
        [
            "",
            "## Local Layout",
            "",
            "- `raw/`: downloaded archives or original dataset files.",
            "- `converted/`: normalized intermediate files.",
            "- `rosbags/`: converted ROS bags for the BEV/VINS evaluation pipeline.",
            "- `metadata/`: sequence lists, calibration notes, and conversion manifests.",
            "",
            "## Recommended Subset",
            "",
        ]
    )
    for item in dataset.get("recommended_subset", []):
        lines.append("- {}".format(item))
    lines.extend(["", "## Notes", "", dataset.get("notes", "")])
    return "\n".join(lines).rstrip() + "\n"


def write_download_plan(registry, output_path):
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Public Dataset Download Plan",
        "",
        "This file records the first-pass public datasets selected for robustness validation.",
        "Large datasets should be downloaded as small subsets first, then converted into ROS bags.",
        "",
    ]
    for dataset in registry.get("datasets", []):
        lines.extend(
            [
                "## {}".format(dataset["name"]),
                "",
                "- id: `{}`".format(dataset["id"]),
                "- role: `{}`".format(dataset.get("role", "")),
                "- homepage: {}".format(dataset.get("homepage", "")),
            ]
        )
        if dataset.get("devkit"):
            lines.append("- devkit: {}".format(dataset["devkit"]))
        if dataset.get("paper"):
            lines.append("- paper: {}".format(dataset["paper"]))
        lines.extend(
            [
                "- local root: `{}`".format(dataset["local_root"]),
                "- modalities: {}".format(", ".join(dataset.get("modalities", []))),
                "- target tasks: {}".format(", ".join(dataset.get("target_tasks", []))),
                "",
                "Recommended first subset:",
            ]
        )
        for item in dataset.get("recommended_subset", []):
            lines.append("- {}".format(item))
        lines.extend(["", "Notes: {}".format(dataset.get("notes", "")), ""])
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output
