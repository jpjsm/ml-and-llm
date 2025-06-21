import pickle

import warnings
warnings.filterwarnings('ignore')

with open("./state/Labels_4-ErrorResults-31x31-v2.pkl", "rb") as inpickle:
    results = pickle.load(inpickle)

top10 = sorted(results.keys())[:10]


print(f"{top10=}")
for errors in top10:
    print(f"{errors=}:")
    print("\t   (nodes_number, epoch_number, batch_size_number, error_count, accuracy, train_delta_secs, test_delta_secs)")
    for scenario in results[errors]:
        print(f"\t=> {scenario}")

    print()