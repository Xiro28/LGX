import json
import sys


def create_dataset_from_file(dataset_path : str, problems : list[str] = None):
    """Reads a JSON file and returns its content as a Python object."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    new_dataset = []
    for entry in data:
        entry_prompt = entry.get("text", "")
        entry_ground_truth = entry.get("output", [])
        entry_problem = entry.get("problem_name", "")

        if problems is not None and entry_problem not in problems:
            continue

        new_dataset.append({
            "problem_name": entry_problem,
            "text": entry_prompt,
            "output": entry_ground_truth
        })
        
    return new_dataset

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python dataset_filter.py <dataset_path> <output_file> <problem1> <problem2> ...")

    dataset_path = sys.argv[1]
    output_file = sys.argv[2]
    problems = sys.argv[3:]

    final_data = create_dataset_from_file(dataset_path, problems)
    
    json_output = json.dumps(final_data, indent=4)

    with open(output_file, "w") as f:
        f.write(json_output)