import os
import json
from collections import Counter

run_dir = input("Enter the path to the directory of the run you want to sub-cluster \n"
                "e.g., experiments/debug/2024-03-10-14-44-29/iteration-4 \nEnter Here: ")
final_path = os.path.join(run_dir, 'final.json')

with open(final_path, 'r') as file:
    data = json.load(file)

descriptions = data['descriptions']
cluster_assignments = data['cluster_predictions']
count = Counter(cluster_assignments)

print("Available clusters and their indices:")
for i, desc in enumerate(descriptions):
    percentage = (int)((count[i] / len(cluster_assignments))*100)
    print(f"Index {i}: {desc}  ({percentage}%)")

indices = input("Enter the indices of clusters you want to sub-cluster (separated by space): ")
cluster_indices = list(map(int, indices.split()))

args_path = os.path.join(run_dir, '../args.json')
with open(args_path, 'r') as file:
    args_data = json.load(file)

data_path = os.path.join(args_data['data_path'], "data.json")
with open(data_path, 'r') as file:
    raw_data = json.load(file)

commands = []  # Store all commands to be run

for index in cluster_indices:
    indices_to_extract = [i for i, j in enumerate(cluster_assignments) if j == index]
    new_data = [raw_data['texts'][i] for i in indices_to_extract]

    # Create dict with specific GoalEx data structure
    export_data = {
        "goal": f"I would like to cluster them based on topics; each cluster should have a description of '<topic>'. \
            Keep in mind that all these texts are already part of a broader cluster of: {descriptions[index]}",
        "texts": new_data,
        "example_descriptions": []
    }

    subset_dir_path = os.path.join(args_data['data_path'], f'subcluster_{index}')
    os.makedirs(subset_dir_path, exist_ok=True)

    subset_data_path = os.path.join(subset_dir_path, 'data.json')
    with open(subset_data_path, 'w') as file:
        json.dump(export_data, file)

    new_exp_dir = os.path.join(run_dir, f'subcluster_{index}')
    os.makedirs(new_exp_dir, exist_ok=True)

    # Command to be run for this subcluster
    cmd = f"CUDA_VISIBLE_DEVICES=2 python src/iterative_cluster.py --data_path {subset_dir_path} " \
          f"--exp_dir {new_exp_dir} --proposer_model gpt-4 --assigner_name google/flan-t5-xl " \
          "--proposer_num_descriptions_to_propose 20 --assigner_for_final_assignment_template templates/t5_multi_assigner_one_output.txt " \
          "--cluster_num_clusters 5 --cluster_overlap_penalty 0.2 --cluster_not_cover_penalty 1.0 " \
          "--iterative_max_rounds 1 --verbose"

    commands.append(cmd)

# Output all commands
for command in commands:
    print(command)