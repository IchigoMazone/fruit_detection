from __future__ import annotations

import argparse
import ast
import hashlib
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FRUIT_ALIASES = {
    "apple": ("apple",),
    "banana": ("banana",),
    "guava": ("guava",),
    "mango": ("mango", "mangoes"),
    "orange": ("orange",),
    "tomato": ("tomato",),
}


@dataclass(frozen=True)
class SourceZip:
    zip_path: Path
    class_name: str


@dataclass(frozen=True)
class ImageRecord:
    class_name: str
    class_id: int
    zip_path: Path
    image_member: str
    labels: tuple[str, ...]

    @property
    def bbox_count(self) -> int:
        return len(self.labels)


def normalize_member_name(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def is_image_member(name: str) -> bool:
    normalized = normalize_member_name(name)
    return Path(normalized).suffix.lower() in IMAGE_EXTENSIONS and "/images/" in normalized


def label_member_for(image_member: str) -> str:
    normalized = normalize_member_name(image_member)
    return str(Path(normalized.replace("/images/", "/labels/")).with_suffix(".txt"))


def read_first_data_yaml(zip_file: ZipFile) -> str:
    for name in zip_file.namelist():
        if Path(normalize_member_name(name)).name in {"data.yaml", "data.yml"}:
            return zip_file.read(name).decode("utf-8", errors="ignore")
    return ""


def parse_names_from_yaml_text(yaml_text: str) -> list[str]:
    match = re.search(r"(?m)^names:\s*(\[.*?\])\s*$", yaml_text)
    if match:
        try:
            names = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            names = []
        if isinstance(names, list):
            return [str(name) for name in names]

    names: list[str] = []
    in_names = False
    for line in yaml_text.splitlines():
        if re.match(r"^names:\s*$", line):
            in_names = True
            continue
        if in_names:
            match = re.match(r"^\s*\d+\s*:\s*(.+?)\s*$", line)
            if match:
                names.append(match.group(1).strip().strip("'\""))
            elif line and not line.startswith(" "):
                break
    return names


def infer_class_name(zip_path: Path) -> str | None:
    with ZipFile(zip_path) as zip_file:
        yaml_text = read_first_data_yaml(zip_file)
    names = parse_names_from_yaml_text(yaml_text)
    searchable = " ".join([zip_path.stem, yaml_text, *names]).lower()

    for class_name, aliases in FRUIT_ALIASES.items():
        if any(alias in searchable for alias in aliases):
            return class_name

    return None


def discover_sources(data_dir: Path) -> list[SourceZip]:
    sources: list[SourceZip] = []
    unknown: list[Path] = []

    for zip_path in sorted(data_dir.glob("*.zip")):
        class_name = infer_class_name(zip_path)
        if class_name is None:
            unknown.append(zip_path)
            continue
        sources.append(SourceZip(zip_path=zip_path, class_name=class_name))

    if unknown:
        names = ", ".join(path.name for path in unknown)
        raise ValueError(f"Could not infer fruit class for: {names}")

    if not sources:
        raise FileNotFoundError(f"No .zip files found in {data_dir}")

    return sources


def read_label_lines(zip_file: ZipFile, label_member: str, target_class_id: int) -> tuple[str, ...]:
    try:
        raw = zip_file.read(label_member).decode("utf-8", errors="ignore")
    except KeyError:
        return ()

    lines: list[str] = []
    for raw_line in raw.splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue

        try:
            x_center, y_center, width, height = [float(value) for value in parts[1:5]]
        except ValueError:
            continue

        if width <= 0 or height <= 0:
            continue

        values = [min(max(value, 0.0), 1.0) for value in (x_center, y_center, width, height)]
        lines.append(
            f"{target_class_id} "
            f"{values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}"
        )

    return tuple(lines)


def collect_records(sources: list[SourceZip], class_names: list[str]) -> dict[str, list[ImageRecord]]:
    records_by_class = {class_name: [] for class_name in class_names}

    for source in sources:
        class_id = class_names.index(source.class_name)
        with ZipFile(source.zip_path) as zip_file:
            image_members = sorted({normalize_member_name(name) for name in zip_file.namelist() if is_image_member(name)})
            for image_member in image_members:
                labels = read_label_lines(zip_file, label_member_for(image_member), class_id)
                records_by_class[source.class_name].append(
                    ImageRecord(
                        class_name=source.class_name,
                        class_id=class_id,
                        zip_path=source.zip_path,
                        image_member=image_member,
                        labels=labels,
                    )
                )

    return records_by_class


def select_records(records: list[ImageRecord], max_images: int | None) -> list[ImageRecord]:
    if max_images is None or len(records) <= max_images:
        return sorted(records, key=lambda record: (record.zip_path.name, record.image_member))

    ranked = sorted(records, key=lambda record: (-record.bbox_count, record.zip_path.name, record.image_member))
    return ranked[:max_images]


def split_records(records: list[ImageRecord], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[ImageRecord]]:
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = round(total * train_ratio)
    val_count = round(total * val_ratio)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def safe_output_stem(record: ImageRecord) -> str:
    digest_source = f"{record.zip_path.name}/{record.image_member}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]
    source_stem = re.sub(r"[^A-Za-z0-9_]+", "_", record.zip_path.stem)
    image_stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(record.image_member).stem)
    source_stem = source_stem[:36].strip("_")
    image_stem = image_stem[:80].strip("_")
    return f"{record.class_name}_{source_stem}_{image_stem}_{digest}"


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_splits(output_dir: Path, splits: dict[str, list[ImageRecord]]) -> None:
    records_by_zip: dict[Path, list[tuple[str, ImageRecord]]] = {}
    for split, records in splits.items():
        for record in records:
            records_by_zip.setdefault(record.zip_path, []).append((split, record))

    for zip_path, split_records in records_by_zip.items():
        with ZipFile(zip_path) as zip_file:
            for split, record in split_records:
                stem = safe_output_stem(record)
                suffix = Path(record.image_member).suffix.lower() or ".jpg"
                image_path = output_dir / "images" / split / f"{stem}{suffix}"
                label_path = output_dir / "labels" / split / f"{stem}.txt"

                with zip_file.open(record.image_member) as source, image_path.open("wb") as target:
                    shutil.copyfileobj(source, target)

                text = "\n".join(record.labels)
                label_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    yaml_path = output_dir / "data.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                f"nc: {len(class_names)}",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def build_dataset(
    data_dir: Path,
    output_dir: Path,
    max_images_per_class: int | None,
    split_ratio: tuple[float, float, float],
    seed: int,
) -> dict[str, dict[str, int | str]]:
    train_ratio, val_ratio, test_ratio = split_ratio
    ratio_total = train_ratio + val_ratio + test_ratio
    if ratio_total <= 0:
        raise ValueError("Split ratio total must be positive")

    train_ratio /= ratio_total
    val_ratio /= ratio_total

    sources = discover_sources(data_dir)
    class_names = sorted({source.class_name for source in sources})
    records_by_class = collect_records(sources, class_names)
    prepare_output_dir(output_dir)

    summary: dict[str, dict[str, int | str]] = {}
    all_splits = {"train": [], "val": [], "test": []}

    for class_name in class_names:
        records = records_by_class[class_name]
        selected = select_records(records, max_images_per_class)
        splits = split_records(selected, train_ratio, val_ratio, seed + class_names.index(class_name))

        for split, records_in_split in splits.items():
            all_splits[split].extend(records_in_split)

        summary[class_name] = {
            "sources": sum(1 for source in sources if source.class_name == class_name),
            "original": len(records),
            "kept": len(selected),
            "removed": len(records) - len(selected),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "bbox": sum(record.bbox_count for record in selected),
        }

    write_splits(output_dir, all_splits)
    yaml_path = write_data_yaml(output_dir, class_names)

    summary["all"] = {
        "classes": len(class_names),
        "sources": len(sources),
        "train": len(all_splits["train"]),
        "val": len(all_splits["val"]),
        "test": len(all_splits["test"]),
        "yaml": str(yaml_path),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge all Roboflow YOLO zip files into one grouped YOLO detect dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/fruit_yolo11"))
    parser.add_argument("--max-images-per-class", type=int, default=0, help="0 means keep every image.")
    parser.add_argument("--split", nargs=3, type=float, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_images = args.max_images_per_class if args.max_images_per_class > 0 else None
    summary = build_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_images_per_class=max_images,
        split_ratio=tuple(args.split),
        seed=args.seed,
    )

    print("Dataset built:")
    for class_name, item in summary.items():
        if class_name == "all":
            continue
        print(
            f"  {class_name}: sources={item['sources']} original={item['original']} "
            f"kept={item['kept']} removed={item['removed']} train={item['train']} "
            f"val={item['val']} test={item['test']} bbox={item['bbox']}"
        )
    print(
        f"  total: classes={summary['all']['classes']} sources={summary['all']['sources']} "
        f"train={summary['all']['train']} val={summary['all']['val']} test={summary['all']['test']}"
    )
    print(f"  data yaml: {summary['all']['yaml']}")


if __name__ == "__main__":
    main()
