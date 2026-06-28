from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")

ARCHIVE_CLASS_MAP = {
    "freshapples": "apple_fresh",
    "freshbanana": "banana_fresh",
    "freshoranges": "orange_fresh",
    "rottenapples": "apple_rotten",
    "rottenbanana": "banana_rotten",
    "rottenoranges": "orange_rotten",
    "unripe apple": "apple_unripe",
    "unripe banana": "banana_unripe",
    "unripe orange": "orange_unripe",
}

TOMATO_CLASS_MAP = {
    "Bad": "tomato_rotten",
    "Green": "tomato_unripe",
    "Red": "tomato_fresh",
}

CLASS_NAMES = [
    "apple_fresh",
    "apple_rotten",
    "apple_unripe",
    "banana_fresh",
    "banana_rotten",
    "banana_unripe",
    "orange_fresh",
    "orange_rotten",
    "orange_unripe",
    "tomato_fresh",
    "tomato_rotten",
    "tomato_unripe",
]


@dataclass(frozen=True)
class ImageRecord:
    zip_path: Path
    member: str
    source_split: str
    source_class: str
    class_name: str


def normalize_member_name(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def is_image_member(name: str) -> bool:
    return Path(normalize_member_name(name)).suffix.lower() in IMAGE_EXTENSIONS


def clean_stem(value: str, limit: int = 96) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return cleaned[:limit].strip("_") or "image"


def output_name(record: ImageRecord) -> str:
    digest_source = f"{record.zip_path.name}/{record.member}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]
    source = clean_stem(record.zip_path.stem, limit=32)
    stem = clean_stem(Path(record.member).stem)
    suffix = Path(record.member).suffix.lower() or ".jpg"
    return f"{record.class_name}_{source}_{stem}_{digest}{suffix}"


def discover_archive_records(zip_path: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with ZipFile(zip_path) as zip_file:
        for member in sorted(zip_file.namelist()):
            normalized = normalize_member_name(member)
            if not is_image_member(normalized):
                continue

            parts = normalized.split("/")
            for index, part in enumerate(parts):
                if part not in {"train", "test"}:
                    continue
                if index + 1 >= len(parts):
                    continue

                source_class = parts[index + 1]
                class_name = ARCHIVE_CLASS_MAP.get(source_class)
                if class_name is None:
                    continue

                records.append(
                    ImageRecord(
                        zip_path=zip_path,
                        member=normalized,
                        source_split=part,
                        source_class=source_class,
                        class_name=class_name,
                    )
                )
                break
    return records


def discover_tomato_records(zip_path: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with ZipFile(zip_path) as zip_file:
        for member in sorted(zip_file.namelist()):
            normalized = normalize_member_name(member)
            if not is_image_member(normalized):
                continue

            parts = normalized.split("/")
            if len(parts) < 3:
                continue

            source_split, source_class = parts[0], parts[1]
            if source_split == "valid":
                source_split = "val"
            if source_split not in SPLITS:
                continue

            class_name = TOMATO_CLASS_MAP.get(source_class)
            if class_name is None:
                continue

            records.append(
                ImageRecord(
                    zip_path=zip_path,
                    member=normalized,
                    source_split=source_split,
                    source_class=source_class,
                    class_name=class_name,
                )
            )
    return records


def assign_balanced_splits(
    records: list[ImageRecord],
    split_ratio: tuple[float, float, float],
    seed: int,
    balance_per_class: bool,
) -> list[tuple[str, ImageRecord]]:
    train_ratio, val_ratio, test_ratio = split_ratio
    ratio_total = train_ratio + val_ratio + test_ratio
    if ratio_total <= 0:
        raise ValueError("split ratio total must be positive")

    train_ratio /= ratio_total
    val_ratio /= ratio_total

    by_class: dict[str, list[ImageRecord]] = {class_name: [] for class_name in CLASS_NAMES}
    for record in records:
        by_class[record.class_name].append(record)

    if any(not class_records for class_records in by_class.values()):
        missing = [class_name for class_name, class_records in by_class.items() if not class_records]
        raise ValueError(f"No images found for classes: {', '.join(missing)}")

    target_per_class = min(len(class_records) for class_records in by_class.values()) if balance_per_class else None
    assigned: list[tuple[str, ImageRecord]] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        shuffled = by_class[class_name][:]
        random.Random(seed + class_index).shuffle(shuffled)
        if target_per_class is not None:
            shuffled = shuffled[:target_per_class]

        train_count = int(len(shuffled) * train_ratio)
        val_count = int(len(shuffled) * val_ratio)
        train_records = shuffled[:train_count]
        val_records = shuffled[train_count : train_count + val_count]
        test_records = shuffled[train_count + val_count :]

        assigned.extend(("train", record) for record in train_records)
        assigned.extend(("val", record) for record in val_records)
        assigned.extend(("test", record) for record in test_records)

    return assigned


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in SPLITS:
        for class_name in CLASS_NAMES:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def write_dataset(output_dir: Path, assigned_records: list[tuple[str, ImageRecord]]) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    by_zip: dict[Path, list[tuple[str, ImageRecord]]] = {}
    for split, record in assigned_records:
        by_zip.setdefault(record.zip_path, []).append((split, record))

    for zip_path, records in by_zip.items():
        with ZipFile(zip_path) as zip_file:
            for split, record in records:
                filename = output_name(record)
                relative_path = Path(split) / record.class_name / filename
                output_path = output_dir / relative_path
                with zip_file.open(record.member) as source, output_path.open("wb") as target:
                    shutil.copyfileobj(source, target)

                manifest.append(
                    {
                        "split": split,
                        "class_name": record.class_name,
                        "class_id": str(CLASS_NAMES.index(record.class_name)),
                        "relative_path": relative_path.as_posix(),
                        "source_zip": record.zip_path.name,
                        "source_split": record.source_split,
                        "source_class": record.source_class,
                        "source_member": record.member,
                    }
                )

    return sorted(manifest, key=lambda row: (row["split"], row["class_name"], row["relative_path"]))


def write_metadata(output_dir: Path, manifest: list[dict[str, str]]) -> None:
    (output_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    names = "\n".join(f"  {index}: {class_name}" for index, class_name in enumerate(CLASS_NAMES))
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: train",
                "val: val",
                "test: test",
                "",
                f"nc: {len(CLASS_NAMES)}",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "class_name",
                "class_id",
                "relative_path",
                "source_zip",
                "source_split",
                "source_class",
                "source_member",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)


def summarize(manifest: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    summary = {class_name: {split: 0 for split in SPLITS} for class_name in CLASS_NAMES}
    for row in manifest:
        summary[row["class_name"]][row["split"]] += 1
    return summary


def build_dataset(
    archive_zip: Path,
    tomato_zip: Path,
    output_dir: Path,
    split_ratio: tuple[float, float, float],
    seed: int,
    balance_per_class: bool,
) -> dict[str, dict[str, int]]:
    records = discover_archive_records(archive_zip) + discover_tomato_records(tomato_zip)
    prepare_output_dir(output_dir)
    assigned_records = assign_balanced_splits(records, split_ratio, seed, balance_per_class)
    manifest = write_dataset(output_dir, assigned_records)
    write_metadata(output_dir, manifest)
    return summarize(manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean ImageFolder fruit quality classification dataset.")
    parser.add_argument("--archive-zip", type=Path, default=Path("data/archive (1).zip"))
    parser.add_argument("--tomato-zip", type=Path, default=Path("data/Tomato Sorter 7K.v1i.folder.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/fruit_quality_classification"))
    parser.add_argument("--split", nargs=3, type=float, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--no-balance-per-class", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        archive_zip=args.archive_zip,
        tomato_zip=args.tomato_zip,
        output_dir=args.output_dir,
        split_ratio=tuple(args.split),
        seed=args.seed,
        balance_per_class=not args.no_balance_per_class,
    )

    print("Dataset built:")
    totals = {split: 0 for split in SPLITS}
    for class_name in CLASS_NAMES:
        counts = summary[class_name]
        for split, count in counts.items():
            totals[split] += count
        print(f"  {class_name}: train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"  total: train={totals['train']} val={totals['val']} test={totals['test']}")
    print(f"  output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
