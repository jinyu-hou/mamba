import pandas as pd
from os import listdir
from os.path import join, dirname
from pathlib import Path
import json

results_dirpath = dirname(__file__) + "/results"
results_files = [f for f in listdir(results_dirpath) if Path(join(results_dirpath, f)).is_file()]
results_files = sorted(results_files, key=len)
results_dict = {}

for fname in results_files:
    if not "-" in fname:
        continue
    fname_split = Path(fname).stem.split("-")
    model_dict = results_dict.setdefault(fname_split[0], {})
    scores_dict = model_dict.setdefault(int(fname_split[1]), {})
    # print(fname)
    if len(fname_split) == 3 and fname_split[-1] == "mmlu":
        with open(join(results_dirpath, fname)) as f:
            lines = json.load(f)
            scores_dict["mmlu"] = lines["results"]["mmlu"]["acc,none"]
    elif len(fname_split) == 3 and fname_split[-1] == "numel":
        with open(join(results_dirpath, fname)) as f:
            lines = json.load(f)
            scores_dict["numel"] = lines["numel"]
    else:
        with open(join(results_dirpath, fname)) as f:
            lines = json.load(f)
            for sname, sdict in lines["results"].items():
                if "acc_norm,none" in sdict:
                    scores_dict[sname] = sdict["acc_norm,none"]
                else:
                    scores_dict[sname] = sdict["acc,none"]

# print(results_dict)
for model, results in results_dict.items():
    df = pd.DataFrame.from_dict(results, orient="index")
    df = df.sort_index(ascending=False).rename_axis('preserve_rate', axis=0).round(4)
    df.to_csv("{}/{}.csv".format(results_dirpath, model))
    

            