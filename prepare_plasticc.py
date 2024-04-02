import urllib.request
import os.path
import pandas as pd
import numpy as np
import json

from tokenizer import LCTokenizer


class_keys = {
    6: 0,
    15: 1,
    16: 2,
    42: 3,
    52: 4,
    53: 5,
    62: 6,
    64: 7,
    65: 8,
    67: 9,
    88: 10,
    90: 11,
    92: 12,
    95: 13,
    991: 14,
    992: 15,
    993: 16,
    994: 17,
}

config = {
    "num_bins": 500,
    "num_time_bins": 500,
    "min_SN": 3,
    "augment_factor": 1,
    "max_delta_time": 1000,
    "min_flux": -10000,
    "max_flux": 10000,
    "static_features": ["hostgal_photoz", "hostgal_photoz_err"],
    "num_labels": 18,
    "class_keys": class_keys,
    "bands": [0, 1, 2, 3, 4, 5],
}

files = [
    "plasticc_train_metadata.csv.gz",
    "plasticc_train_lightcurves.csv.gz",
    "plasticc_test_metadata.csv.gz",
]

test_files = [
    "plasticc_test_lightcurves_01.csv.gz",
    "plasticc_test_lightcurves_02.csv.gz",
    "plasticc_test_lightcurves_03.csv.gz",
    "plasticc_test_lightcurves_04.csv.gz",
    "plasticc_test_lightcurves_05.csv.gz",
    "plasticc_test_lightcurves_06.csv.gz",
    "plasticc_test_lightcurves_07.csv.gz",
    "plasticc_test_lightcurves_08.csv.gz",
    "plasticc_test_lightcurves_09.csv.gz",
    "plasticc_test_lightcurves_10.csv.gz",
    "plasticc_test_lightcurves_11.csv.gz",
]

if not os.path.exists("plasticc"):
    os.makedirs("plasticc")

for file in files + test_files:
    if not os.path.isfile(os.path.join("plasticc", file)):
        urllib.request.urlretrieve("https://zenodo.org/record/2539456/files/%s" % file, os.path.join("plasticc", file))

df_train_meta = pd.read_csv("plasticc/plasticc_train_metadata.csv.gz")
df_train = pd.read_csv("plasticc/plasticc_train_lightcurves.csv.gz")
print(np.percentile(df_train["flux"], [0.1, 99.9]))

df_test_meta = pd.read_csv("plasticc/plasticc_test_metadata.csv.gz")
for file in test_files:
    df_test = pd.read_csv(os.path.join("plasticc", file))
    print(file, np.percentile(df_test["flux"], [0.1, 99.9]))

tokenizer = LCTokenizer(config["min_flux"], config["max_flux"], config["num_bins"], config["max_delta_time"],
                        config["num_time_bins"], bands=config["bands"],
                        transform=np.arcsinh, inverse_transform=np.sinh,
                        min_SN=config["min_SN"])

config["vocab_size"] = tokenizer.vocab_size

with open("plasticc/dataset_config.json", "w") as f:
    json.dump(config, f)


def load_sequences(df_meta, df, augment_factor=1):
    sequences = []
    token_augs = []
    for i in range(augment_factor):
        token_augs.append(tokenizer.encode(df))
    zipped = [df_meta["object_id"], df_meta["true_target"]]
    for static_feature in config["static_features"]:
        if static_feature in df_meta.columns:
            zipped += [df_meta[static_feature]]
    for row in zip(*zipped):
        id = row[0]
        class_label = class_keys[int(row[1])]
        static = list(row[2:])
        for i in range(augment_factor):
            if len(token_augs[i][id]) >= 2:
                sequences.append({"x": token_augs[i][id], "class": class_label, "static": static, "object_id": id})
    return sequences


train_sequences = load_sequences(df_train_meta, df_train, augment_factor=config["augment_factor"])
np.save("plasticc/train.npy", train_sequences)

print("Num train:", len(train_sequences))

test_sequences = []
for i, file in enumerate(test_files):
        df_test = pd.read_csv(os.path.join("plasticc", file))
        test_sequences += load_sequences(df_test_meta[df_test_meta["object_id"].isin(df_test["object_id"].values)],
                                         df_test)
np.save("plasticc/test.npy", test_sequences)

print("Num test:", len(test_sequences))
