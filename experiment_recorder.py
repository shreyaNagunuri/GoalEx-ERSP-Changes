import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import utils_performance
import random


class ExperimentRecorder:
    def __init__(self, problem, label=None):
        self.problem = problem
        self.label = label
        self.n = len(problem.texts)

        self.output_dir = None
        self.iteration = 0

    def record_propose(self, descriptions, name):
        with open(
            os.path.join(self.output_dir, f"iteration-{self.iteration}", f"{name}.json"),
            "w",
        ) as f:
            json.dump(
                {
                    "descriptions": descriptions,
                },
                f,
            )

    def record_assign(self, descriptions, text_descriptions_matching, name):
        print("Stage Assign", name)
        (
            all_recall,
            all_precision,
            single_recalls,
            single_precisions,
        ) = utils_performance.get_descriptions_performance(text_descriptions_matching)
        print("Assigned all recall:", all_recall)
        print("Assigned all precision:", all_precision)
        for description, recall, precision in zip(
            descriptions, single_recalls, single_precisions
        ):
            print(
                f"Assigned description: {description}. recall: {recall} precision: {precision}"
            )

        with open(
            os.path.join(self.output_dir, f"iteration-{self.iteration}", f"{name}.json"),
            "w",
        ) as f:
            json.dump(
                {
                    "descriptions": descriptions,
                    "text_descriptions_matching": text_descriptions_matching.tolist(),
                },
                f,
            )

    def record_select(self, descriptions, cluster_predictions, name):
        print("Stage select", name)
        distribution = Counter(cluster_predictions)
        print("Unmatched count:", distribution[-1])
        for i in range(len(descriptions)):
            print(f"Description: {descriptions[i]}. Count: {distribution[i]}")

        if self.label is not None:
            labels = np.array(self.label.labels)
            unmatched_text_indices = cluster_predictions == -1
            (
                normalized_mutial_info,
                adjusted_rand_index,
                macro_f1,
            ) = utils_performance.get_cluster_performance(
                labels[~unmatched_text_indices],
                cluster_predictions[~unmatched_text_indices],
            )

            print(
                "On matched texts:",
                normalized_mutial_info,
                adjusted_rand_index,
                macro_f1,
            )
            true_descriptions = self.label.class_descriptions
            _, mapping = utils_performance.assign_labels(
                labels[~unmatched_text_indices],
                cluster_predictions[~unmatched_text_indices],
            )
            mapped_descriptions = []
            for i in range(len(true_descriptions)):
                mapped_descriptions.append(
                    [
                        descriptions[p]
                        for p in range(len(descriptions))
                        if mapping.get(p, -1) == i
                    ]
                )
                print(
                    f"True description: {true_descriptions[i]}",
                    "|||",
                    "Mapped descriptions:",
                    mapped_descriptions[i],
                )

            with open(
                os.path.join(
                    self.output_dir, f"iteration-{self.iteration}", f"{name}_results.json"
                ),
                "w",
            ) as f:
                json.dump(
                    {
                        "normalized_mutial_info": normalized_mutial_info,
                        "adjusted_rand_index": adjusted_rand_index,
                        "macro_f1": macro_f1,
                        "mapped_descriptions": mapped_descriptions,
                        "num_unmatched_text_indices": np.sum(
                            unmatched_text_indices
                        ).item(),
                    },
                    f,
                )

        with open(
            os.path.join(self.output_dir, f"iteration-{self.iteration}", f"{name}.json"),
            "w",
        ) as f:
            json.dump(
                {
                    "descriptions": descriptions,
                    "cluster_predictions": cluster_predictions.tolist(),
                },
                f,
            )
        
        self.generate_cluster_summary(descriptions=descriptions, cluster_predictions=cluster_predictions, name=name)

#ERSP CHANGES START HERE
    #CLUSTER SUMMARIES
    def generate_cluster_summary(self, descriptions, cluster_predictions, name):
        texts = self.problem.texts
        total_predictions = self.n
        distribution = Counter(cluster_predictions)
        #cluster_info will be a list of dicts, where each dict describes a cluster
        cluster_info = []
        cluster_description = []
        cluster_percentage = []

        #i is the index, desc is the value at that index
        for i, desc in enumerate(descriptions):
            count = distribution[i]
            percentage = (count / total_predictions) * 100 if total_predictions!=0 else 0
            cluster_texts = [texts[j] for j, cluster_num in enumerate(cluster_predictions) if cluster_num == i]
            examples = random.sample(cluster_texts, min(3, len(cluster_texts)))
            cluster_description.append(desc)
            cluster_percentage.append(percentage)
            cluster_info.append({
                'description': desc,
                'count': count,
                'percentage': percentage,
                'examples': examples
            })
            
        filename = os.path.join(self.output_dir, f"iteration-{self.iteration}", f"{name}_cluster_info.txt")

        with open(filename, 'w') as file:
            file.write("This text file provides more information about each selected cluster\n" + "-" * 70 + "\n")
            file.write(f"Unmatched count: {distribution[-1]} = {(distribution[-1]/total_predictions):.2f}%   Data size: {total_predictions}\n" + "-" * 70 + "\n")
            for info in cluster_info:
                file.write(f"==> Description: {info['description']}\n")
                file.write(f"==> Count: {info['count']}\n")
                file.write(f"==> Percentage: {info['percentage']:.2f}%\n")
                file.write("==> Examples:\n")
                for example in info['examples']:
                    file.write(f"- {example}\n")
                file.write("\n" + "-" * 70 + "\n")  # adding a line of dashes for separation
        
        # Creating plot (VISUALIZATIONS)
        fig = plt.figure(figsize=(20, 10))
        plt.suptitle('Cluster Distribution')
        plt.pie(cluster_percentage, labels=cluster_description, autopct='%1.1f%%')
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, f"iteration-{self.iteration}", "plot.png")
        plt.savefig(fig_path)
        # show plot
        plt.show()
    #ERSP CHANGES END HERE
    def next_iteration(self):
        self.iteration += 1
        os.makedirs(
            os.path.join(self.output_dir, f"iteration-{self.iteration}"), exist_ok=True
        )

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(
            os.path.join(self.output_dir, f"iteration-{self.iteration}"), exist_ok=True
        )
