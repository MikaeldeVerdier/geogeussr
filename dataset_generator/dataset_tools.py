import os
import shutil

from dataset_generator.files import save_annotations, load_annotations

def combine_dirs(*input_dirs, output_dir):  # TODO: instead of this, just copy the entire dir and then remove the ones that aren't in the annotations (perhaps only if clean is desired)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    new_annotations = []
    for input_dir in input_dirs:
        old_annotations = load_annotations(input_dir)

        for old_annotation in old_annotations:
            old_path = os.path.join(input_dir, old_annotation["image_name"])

            if not os.path.exists(old_path):
                continue

            new_path = os.path.join(output_dir, old_annotation["image_name"])
            shutil.copyfile(old_path, new_path)

            new_annotations.append(old_annotation)

    save_annotations(new_annotations, output_dir)


def clean_dir(input_dir, output_dir):  # 'clean' here means making sure only annotated images are in the dir
    combine_dirs(input_dir, output_dir=output_dir)
