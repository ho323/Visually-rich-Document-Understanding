import json
import os
from pathlib import Path

import tqdm
import typer
import yaml


def main(root_folder: Path = "./datasets"):
    with open(root_folder / "data.yaml", "w") as f:
        yaml.dump(
            {
                "path": "./",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {
                    "0": "Caption",
                    "1": "Footnote",
                    "2": "Formula",
                    "3": "List-item",
                    "4": "Page-footer",
                    "5": "Page-header",
                    "6": "Picture",
                    "7": "Section-header",
                    "8": "Table",
                    "9": "Text",
                    "10": "Title",
                },
            },
            f,
        )

    for folder in ["val", "test", "train"]:
        print(f"convert {folder} dataset...")
        os.makedirs(root_folder / "labels" / folder, exist_ok=True)
        os.makedirs(root_folder / "images" / folder, exist_ok=True)
        with open(root_folder / "COCO" / f"{folder}.json") as f:
            bigjson = json.load(f)
        for image in tqdm.tqdm(bigjson["images"], desc="move images..."):
            image_id = image["id"]
            filename = image["file_name"]
            os.rename(
                root_folder / "PNG" / filename,
                root_folder / "images" / folder / f"{image_id}.png",
            )
        for annotation in tqdm.tqdm(bigjson["annotations"], desc="write labels..."):
            image_id = annotation["image_id"]
            filename = f"{image_id}.txt"
            x, y, w, h = annotation["bbox"]

            
            img_info = next(img for img in bigjson["images"] if img["id"] == image_id)
            img_w, img_h = img_info["width"], img_info["height"]

            
            x_c = (x + w / 2) / img_w
            y_c = (y + h / 2) / img_h
            nw  = w / img_w
            nh  = h / img_h

            category_id = annotation["category_id"]  
            with open(root_folder / "labels" / folder / filename, "a") as f:
                f.write(f"{category_id} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}\n")


if __name__ == "__main__":
    typer.run(main)
