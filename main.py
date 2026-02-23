from src.lgx import lgx

if __name__ == "__main__":

    lgx_instance = lgx.create(
        "llama3.1:70b",
        "benchmark/behaviours/behaviour_v3.yml",
        "benchmark/applications/ALL/lgx.yml"
    )

    # load the database.json file and iterate over the questions, calling lgx_instance.infer for each question
    with open("dataset_final_lnrs.json", "r") as f:
        import json
        data = json.load(f)
        for item in data:
            prompt = item.get("text")
            result = lgx_instance.infer(prompt).get_extracted_atoms()
            print(result)
            break
