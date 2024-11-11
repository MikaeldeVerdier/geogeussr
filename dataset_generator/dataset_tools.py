import os
import shutil

from dataset_generator.files import save_annotations, load_annotations

def combine_dirs(*input_dirs, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    new_annotations = []
    for input_dir in input_dirs:
        old_annotations = load_annotations(input_dir)

        for old_annotation in old_annotations:
            old_path = old_annotation["image_path"]

            if not os.path.exists(old_path):
                continue

            new_path = os.path.join(output_dir, old_annotation["image_path"].replace("\\", "/").split("/")[-1])
            shutil.copyfile(old_path, new_path)

            new_annotations.append(old_annotation)
            new_annotations[-1]["image_path"] = new_path

    save_annotations(new_annotations, output_dir)


def clean_dir(input_dir, output_dir):  # 'clean' here means making sure only annotated images are in the dir
    combine_dirs(input_dir, output_dir=output_dir)
