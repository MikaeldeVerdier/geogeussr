import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import validation_inference_viz_config as viz_cfg

class ValidationInferenceVisualizer:
    def __init__(self, save_path=viz_cfg.save_path, inference_dir=viz_cfg.inference_dir, strict_grouping=viz_cfg.strict_grouping):
        self.save_path = save_path
        self.strict_grouping = strict_grouping

        self.create_dir(save_path)

        inference_information = self.load_inferences(inference_dir)
        self.prepare_inferences(inference_information)

    def load_inferences(self, inference_dir):
        inference_informations = {}
        for inference_file in os.listdir(inference_dir):
            image_name = inference_file.replace(".json", "")

            inference_path = os.path.join(inference_dir, inference_file)
            with open(inference_path, encoding="utf8") as json_file:  # could have a files.py file for this
                inference_informations[image_name] = json.load(json_file)

        return inference_informations

    def prepare_inferences(self, inference_information):
        # correct_prompts = [information["correct_prompt"] for information in inference_information.values()]
        # half_flattened_info = [information["refinement_results"] for information in inference_information.values()]

        correct_prompts, half_flattened_info = zip(*[(information["correct_prompt"], information["refinement_results"]) for information in inference_information.values()])

        max_refinement = [[len(ref["used_prompt_components"][0]) for ref in refinement] for refinement in half_flattened_info]
        max_refinement_steps = max([len(m) for m in max_refinement])        
        self.refinement_levels = [
            max(
                [m[i] for m in max_refinement if i < len(m)]
            )
            for i in range(max_refinement_steps)
        ]  # np.max(max_refinement, axis=0)

        self.flattened_info = [
            [half_information[i] for half_information in half_flattened_info if i < len(half_information)]
            for i in range(max_refinement_steps)
        ]

        if min(self.refinement_levels) == 2:
            self.flattened_info.insert(0, self.flattened_info[0])  # needs to be inserted because no explicit prompt was done for continent, but can just use which country was chosen

        self.groups = []
        self.group_ns = []
        self.confidence_matrix = []
        for i in range(max(self.refinement_levels)):
            self.groups.append({})
            for i2, corr_prompt in enumerate(correct_prompts):
                correct_region = corr_prompt[0][i]
                self.groups[-1].setdefault(correct_region, []).append(i2)

            grouped_info = {
                k: [
                    self.flattened_info[i][j]
                    for j in v
                    if not self.strict_grouping or
                    (j < len(self.flattened_info[i]) and k in [a[0][i]for a in self.flattened_info[i][j]["prompt_components"]])
                ] for k, v in self.groups[-1].items()
            }

            self.group_ns.append({})
            used_confidences = {}
            for info_group, group_info in grouped_info.items():
                used_confidences[info_group] = {}

                for inference_info in group_info:
                    for prompt_component, confidence in zip(inference_info["prompt_components"], inference_info["confidences"]):
                        if prompt_component[0][i] not in used_confidences[info_group]:
                            used_confidences[info_group][prompt_component[0][i]] = 0

                        used_confidences[info_group][prompt_component[0][i]] += confidence

                sum_confs = sum(used_confidences[info_group].values())  # len(group_info)
                used_confidences[info_group] = {k: v / sum_confs for k, v in used_confidences[info_group].items()}

                self.group_ns[-1][info_group] = len(group_info)

            self.confidence_matrix.append(used_confidences)

    def create_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def get_all_labels(self, confidence_matrix):
        all_labels = []
        for used_confidence in confidence_matrix.values():
            all_labels.extend(list(used_confidence.keys()))

        return sorted(set(all_labels))

    def get_colors(self, all_labels):  # improve this to avoid using same (very similar) color for labels that are next to each other in pie diagram
        base_colors = plt.get_cmap("tab20").colors
        smooth_tab20 = LinearSegmentedColormap.from_list("smooth_tab20", base_colors, N=256)
        # tab20 = plt.get_cmap("tab20")

        color_indices = np.linspace(0, 1, len(all_labels))
        np.random.shuffle(color_indices)
        colors_dict = {
            label: smooth_tab20(color_indices[i])
            for i, label in enumerate(all_labels)
        }

        return colors_dict

    def visualize_pie(self):
        for i in range(max(self.refinement_levels)):
            all_labels = self.get_all_labels(self.confidence_matrix[i])  # set([comps[0][i] for flat_info in self.flattened_info[i] for comps in flat_info["prompt_components"]])
            colors_dict = self.get_colors(all_labels)

            for info_group, used_confidence in self.confidence_matrix[i].items():
                if not len(used_confidence):
                    continue

                confidence_threshold = 0.05
                sorted_confidences_dict = sorted(used_confidence.items(), key=lambda x: x[1], reverse=True)

                sorted_keys = np.array(sorted_confidences_dict)[:, 0]
                sorted_values = np.array(sorted_confidences_dict)[:, 1]

                # colors = {"Asia": "green"}

                filtered_labels = [label if value > confidence_threshold or label == info_group else "" for label, value in sorted_confidences_dict]
                explode = [0.1 if label == info_group else 0 for label in filtered_labels]
                # default_colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
                # default_colors = [color for color in default_colors if color not in colors.values()]
                # used_colors = [colors.get(label, default_colors[i2 % len(default_colors)]) for i2, label in enumerate(filtered_labels)]

                colors = [colors_dict[label] for label in sorted_keys]
                pct_format = lambda x: f"{x:.1f}%" if x > confidence_threshold * 100 else ""
                wedges, texts, autotexts = plt.pie(sorted_values, labels=filtered_labels, autopct=pct_format, colors=colors, explode=explode)  # , colors=used_colors)
                
                for i2, text in enumerate(texts):
                    if filtered_labels[i2] == info_group:  # text.get_text() == info_group:
                        text.set_fontweight("bold")

                plt.title(f"Prediction Confidence Distribution for {info_group}\n({self.group_ns[i][info_group]} {'samples' if self.group_ns[i][info_group] != 1 else 'sample'})")
                plt.tight_layout()

                self.create_dir(os.path.join(self.save_path, "pie"))
                dir_path = os.path.join(self.save_path, "pie", f"r{i}")
                self.create_dir(dir_path)

                validation_inference_path = os.path.join(dir_path, f"pie_{info_group}.png")
                plt.savefig(validation_inference_path)
                plt.close()

    def construct_matrix(self, confidence_dict):
        all_labels = self.get_all_labels(confidence_dict)

        if not len(all_labels):
            return [], []

        confidence_list = []
        for label in all_labels:
            if label not in confidence_dict:
                confidence_list.append([np.NaN for _ in all_labels])

                continue

            confidence_list.append([])
            for label2 in all_labels:
                if label2 not in confidence_dict[label]:
                    confidence_list[-1].append(np.NaN)  # only happens if not strict
                else:
                    confidence_list[-1].append(confidence_dict[label][label2])

        return np.array(confidence_list), all_labels
    
    def strict_construct_matrices(self, confidence_dict):
        groups = {}
        for i, (label, values) in enumerate(confidence_dict.items()):
            key = "".join(list(values.keys()))
            groups.setdefault(key, []).append(label)

        grouped_confidence_dicts = [{i: confidence_dict[i] for i in group} for group in groups.values()]
        confidence_matrices_and_labels = [self.construct_matrix(conf_dict) for conf_dict in grouped_confidence_dicts]

        return zip(*confidence_matrices_and_labels)

    def plot_matrix(self, matrix, labels, matrix_name, group_n_samples, i=0, file_name=""):
        fig_size = int(len(labels) * 0.28 + 8)  # regressed function
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        ax.plot([-0.5, len(labels) - 0.5], [-0.5, len(labels) - 0.5], color="red", lw=2, ls="--")
        # ax.grid(which="major", color="black", linestyle="-", linewidth=0.5)

        cmap = plt.get_cmap("viridis")
        cmap.set_bad(color="black")

        cax = ax.matshow(matrix, cmap=cmap, vmin=0, vmax=1)

        plt.colorbar(cax, shrink=0.8)

        ax.xaxis.tick_top() 
        ax.yaxis.set_label_position("right")

        ax.set_xlabel("Predicted", labelpad=8)
        ax.set_ylabel("Truth", labelpad=16, rotation=270)  # for some reason needs to be padded more

        y_ticks = [f"{label} ({group_n_samples.get(label, 0)} {'samples' if group_n_samples.get(label, 0) != 1 else 'sample'})" for label in labels]  # include group_ns in labels
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="left")
        ax.set_yticks(np.arange(len(y_ticks)), labels=y_ticks)

        ax.set_xticks(np.arange(-0.5, len(labels) - 0.5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels) - 0.5, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

        plt.title(f"Prediction Confidence Matrix for {matrix_name}")

        self.create_dir(os.path.join(self.save_path, "matrix"))
        dir_path = os.path.join(self.save_path, "matrix", f"r{i}")
        self.create_dir(dir_path)

        validation_inference_path = os.path.join(dir_path, file_name)  # weird to use join when file_name just uses / in its string
        plt.savefig(validation_inference_path, bbox_inches="tight", pad_inches=0.5)
        plt.close()
        # ax.clear()

    def visualize_matrix(self):
        for i in range(max(self.refinement_levels)):
            if i == 0:
                matrix_label = "Continents"
            elif i == 1:
                matrix_label = "Countries"

            if self.strict_grouping:
                confidence_matrices, all_labels = self.strict_construct_matrices(self.confidence_matrix[i])
                for confidence_matrix, all_label in zip(confidence_matrices, all_labels):
                    if not len(confidence_matrix):
                        continue

                    if i >= 2:
                        group_labels = [
                            sorted(comp[0][i] for comp in infos["prompt_components"])
                            for infos in self.flattened_info[i]
                        ]  # [sorted(a.keys()) for a in self.confidence_matrix[i].values()]
                        label_indices = [
                            index
                            for index, value in enumerate(group_labels)
                            if all([val in all_label for val in value])
                        ]

                        region_prompt = self.flattened_info[i][label_indices[0]]  # any element in label_indices should work
                        matrix_label = region_prompt["prompt_components"][0][0][i - 1]

                    file_name = f"matrix_{matrix_label}.png"
                    self.plot_matrix(confidence_matrix, all_label, matrix_label, self.group_ns[i], i=i, file_name=file_name)

                continue

            if i == 2:
                matrix_label = "Provinces"
            elif i == 3:
                matrix_label = "Cities"
            elif i > 3:
                matrix_label = "Unknown"  # should never happen

            confidence_matrix, all_labels = self.construct_matrix(self.confidence_matrix[i])
            file_name = f"matrix_{matrix_label}.png"
            self.plot_matrix(confidence_matrix, all_labels, matrix_label, self.group_ns[i], i=i, file_name=file_name)


if __name__ == "__main__":
    visualizer = ValidationInferenceVisualizer()
    visualizer.visualize_pie()
    visualizer.visualize_matrix()
