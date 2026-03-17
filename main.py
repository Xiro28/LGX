from src.lgx import lgx
import tqdm

if __name__ == "__main__":

    lgx_instance = lgx.create(
        "llama3.1:70b",
        "benchmark/behaviours/behaviour_v3.yml",
        "benchmark/applications/LNRS/lgx.yml"
    )

    # load the database.json file and iterate over the questions, calling lgx_instance.infer for each question
    with open("dataset_final_lnrs.json", "r") as f:
        import json
        data = json.load(f)

        with tqdm.tqdm(
            data,
            desc="Instances",
            unit="inst",
            miniters=1,
            mininterval=0
        ) as pbar:
            for item in pbar:
                prompt = item.get("text")
                result = lgx_instance.infer(prompt).get_extracted_atoms()
                #pbar.write(f"Extracted atoms: {result}\n")
                lgx_instance.cleanup()
