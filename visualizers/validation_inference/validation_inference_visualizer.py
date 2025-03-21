import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import validation_inference_viz_config as viz_cfg

class ValidationInferenceVisualizer:
    def __init__(self, save_path=viz_cfg.save_path, inference_dir=viz_cfg.inference_dir):
        self.save_path = save_path

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        inference_information = self.load_inferences(inference_dir)
        self.prepare_inferences(inference_information)

    def load_inferences(self, inference_dir):
        inference_informations = {}
        for inference_file in os.listdir(inference_dir):
            image_name = inference_file.replace(".json", "")

            inference_path = os.path.join(inference_dir, inference_file)
            with open(inference_path, "r") as json_file:
                inference_informations[image_name] = json.load(json_file)

        return inference_informations

    def gc(self, all_labels):
        colormap = plt.get_cmap("tab20")

        color_indices = np.linspace(0, 1, len(all_labels))
        np.random.shuffle(color_indices)
        colors_dict = {
            label: mcolors.to_hex(colormap(color_indices[i]))
            for i, label in enumerate(all_labels)
        }

        return colors_dict

    def prepare_inferences(self, inference_information):
        correct_prompts = [information["correct_prompt"] for information in inference_information.values()]
        half_flattened_info = [information["refinement_results"] for information in inference_information.values()]

        max_refinement = [[len(ref["used_prompt_components"][0]) for ref in refinement] for refinement in half_flattened_info]
        max_refinement_steps = max([len(m) for m in max_refinement])        
        self.refinement_levels = [max([m[i] for m in max_refinement]) for i in range(max_refinement_steps)]  # np.max(max_refinement, axis=0)

        self.flattened_info = [[half_information[i] for half_information in half_flattened_info] for i in range(max_refinement_steps)]

        if min(self.refinement_levels) == 2:
            self.flattened_info.insert(0, self.flattened_info[0])  # needs to be inserted because no explicit prompt was done for continent, but can just use which country was chosen

        self.confidence_matrix = []
        for i in range(max(self.refinement_levels)):
            groups = {}
            for i2, corr_prompt in enumerate(correct_prompts):
                correct_region = corr_prompt[0][i]
                groups.setdefault(correct_region, []).append(i2)

            grouped_info = {k: [self.flattened_info[i][j] for j in v] for k, v in groups.items()}

            used_confidences = {}
            for info_group, group_info in grouped_info.items():
                used_confidences[info_group] = {}

                for inference_info in group_info:
                    for prompt_component, confidence in zip(inference_info["prompt_components"], inference_info["confidences"]):
                        if prompt_component[0][i] not in used_confidences[info_group]:
                            used_confidences[info_group][prompt_component[0][i]] = 0

                        used_confidences[info_group][prompt_component[0][i]] += confidence

                sum_confs = sum(used_confidences[info_group].values())
                used_confidences[info_group] = {k: v / sum_confs for k, v in used_confidences[info_group].items()}

            self.confidence_matrix.append(used_confidences)

    def get_all_labels(self, confidence_matrix):
        all_labels = []
        for used_confidence in confidence_matrix.values():
            all_labels.extend(list(used_confidence.keys()))

        return set(all_labels)

    def visualize_pie(self):
        for i in range(max(self.refinement_levels)):
            all_labels = self.get_all_labels(self.confidence_matrix[i])  # set([comps[0][i] for flat_info in self.flattened_info[i] for comps in flat_info["prompt_components"]])
            colors_dict = self.gc(all_labels)

            for info_group, used_confidence in self.confidence_matrix[i].items():
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

                plt.title(f"Prediction Confidence Distribution for {info_group}")
                plt.tight_layout()

                validation_inference_path = os.path.join(self.save_path, f"validation_inference_pie_{info_group}.png")
                plt.savefig(validation_inference_path)
                plt.close()

    def construct_matrix(self, all_labels, confidence_dict):
        confidence_list = []
        for label in all_labels:
            if label not in confidence_dict:
                confidence_list.append([0 for _ in all_labels])

                continue

            confidence_list.append([])
            for label2 in all_labels:
                if label2 not in confidence_dict[label]:
                    confidence_list[-1].append(0)
                else:
                    confidence_list[-1].append(confidence_dict[label][label2])

        return np.array(confidence_list)

    def visualize_matrix(self):
        for i in range(max(self.refinement_levels)):
            all_labels = sorted(self.get_all_labels(self.confidence_matrix[i]))
            confidence_matrix = self.construct_matrix(all_labels, self.confidence_matrix[i])

            fig_size = int(len(all_labels) * 0.28 + 8)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))

            ax.plot([-0.5, len(all_labels) - 0.5], [-0.5, len(all_labels) - 0.5], color="red", lw=2, ls="--")
            # ax.grid(which="major", color="black", linestyle="-", linewidth=0.5)

            cax = ax.matshow(confidence_matrix)

            plt.colorbar(cax, shrink=0.8)

            ax.xaxis.tick_top()
            ax.yaxis.set_label_position("right")

            ax.set_xlabel("Predicted", labelpad=8)
            ax.set_ylabel("Correct", labelpad=18, rotation=270)  # for some reason needs to be padded more

            ax.set_xticks(np.arange(len(all_labels)), labels=all_labels, rotation=45, ha="left")
            ax.set_yticks(np.arange(len(all_labels)), labels=all_labels)

            # ax.grid(which="minor")

            # ax.set_xticklabels(all_labels, rotation=45, ha="left")
            # ax.set_yticklabels(all_labels)

            ax.set_xticks(np.arange(-0.5, len(all_labels) - 0.5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(all_labels) - 0.5, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

            plt.title("Prediction Confidence Matrix")

            validation_inference_path = os.path.join(self.save_path, f"validation_inference_matrix_r{i}.png")
            plt.savefig(validation_inference_path, bbox_inches="tight", pad_inches=0.5)
            plt.close()
            # ax.clear()


if __name__ == "__main__":
    visualizer = ValidationInferenceVisualizer()
    # visualizer.visualize_pie()
    visualizer.visualize_matrix()
