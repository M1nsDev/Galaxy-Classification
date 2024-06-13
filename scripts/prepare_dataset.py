from collections.abc import Callable
from typing import Any, Optional, cast
from pathlib import Path
from argparse import ArgumentParser, Namespace

import h5py
import pandas as pd

import torch
from torchvision.io import read_image
from torchvision.transforms.v2 import Compose, CenterCrop, Resize


TYPE_DECISION_GRAPH = {
    "smooth-or-featured_smooth_fraction": {
        "how-rounded_round_fraction": 0,
        "how-rounded_in-between_fraction": 1,
        "how-rounded_cigar-shaped_fraction": 2,
    },
    "smooth-or-featured_featured-or-disk_fraction": {
        "edge-on-bulge_rounded_fraction": 3,
        "edge-on-bulge_boxy_fraction": 4,
        "edge-on-bulge_none_fraction": 5,
        "spiral-winding_tight_fraction": 6,
        "spiral-winding_medium_fraction": 7,
        "spiral-winding_loose_fraction": 8,
        "has-spiral-arms_no_fraction": 9,
    },
    "smooth-or-featured_artifact_fraction": None,
}

Transformation = Callable[[torch.Tensor], torch.Tensor]


def main(args: Namespace) -> None:
    image_infos = pd.read_parquet(args.info_file).set_index("iauname", drop=True)
    preprocess = Compose([CenterCrop(207), Resize(69)])

    with h5py.File(args.out_file, "w") as out_file:
        prepare_images_and_classes(
            image_infos=image_infos,
            image_files=recursive_image_search(args.data_dir),
            out_file=out_file,
            final_image_shape=(3, 69, 69),
            preprocess_image=preprocess,
        )


def recursive_image_search(dir: Path) -> list[Path]:
    return list(dir.rglob("*.png"))


def prepare_images_and_classes(
    image_infos: pd.DataFrame,
    image_files: list[Path],
    out_file: h5py.File | h5py.Group,
    final_image_shape: tuple[int, int, int],
    preprocess_image: Optional[Transformation],
) -> None:
    images = out_file.create_dataset(
        name="images",
        shape=(0, *final_image_shape),
        maxshape=(None, *final_image_shape),
        dtype="uint8",
    )
    classes = out_file.create_dataset(
        name="classes",
        shape=(0,),
        maxshape=(None,),
        dtype="uint8",
    )

    for img_file in image_files:
        img_name = img_file.stem

        try:
            attribute_probs = cast(pd.Series, image_infos.loc[img_name])
            galaxy_type = galaxy_type_decider(attribute_probs)

            if galaxy_type is None:
                print(
                    f"[-] Unable to determine specific class for image {img_name} ({img_file})."
                )
                continue

            image = read_image(str(img_file))
            if preprocess_image is not None:
                image = preprocess_image(image)

            append(images, image.numpy())
            append(classes, galaxy_type)
            print(f"[+] Information for image {img_name} ({img_file}) found and saved.")

        except KeyError:
            print(f"[-] No information for image {img_name} ({img_file}).")


def append(dataset: h5py.Dataset, value: Any) -> None:
    dataset.resize(dataset.shape[0] + 1, axis=0)
    dataset[-1] = value


def galaxy_type_decider(attribute_probs: pd.Series) -> int | None:
    subgraph: dict[str, dict[str, int] | None] | dict[str, int] = TYPE_DECISION_GRAPH

    while True:
        current_attributes = list(subgraph.keys())
        max_prob_attribute = cast(str, attribute_probs[current_attributes].idxmax())

        if attribute_probs[max_prob_attribute] < 0.75:
            return None

        next_subgraph = subgraph[max_prob_attribute]

        if isinstance(next_subgraph, int) or next_subgraph is None:
            return next_subgraph

        subgraph = next_subgraph


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default="./data/images/gz_decals/")
    parser.add_argument(
        "--info-file", type=Path, default="./data/gzDecals_auto_posteriors.parquet"
    )
    parser.add_argument("--out-file", type=Path, default="./data/gzDecals_galaxy_zoo_0-75.h5")
    args = parser.parse_args()

    main(args)