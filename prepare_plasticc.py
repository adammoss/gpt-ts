import urllib.request
import os.path
import pandas as pd
import numpy as np
import json
import argparse
import time

from tokenizer import LCTokenizer
from gaussian_process import fit_2d_gp

from sklearn.model_selection import train_test_split

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

class_names = {
    0: "\mu",
    1: "TDE",
    2: "EBE",
    3: "SNII",
    4: "SNIax",
    5: "MIRA",
    6: "SNIbc",
    7: "KN",
    8: "M-dwarf",
    9: "SNIa-91bg",
    10: "AGN",
    11: "SNIa",
    12: "RR Lyrae",
    13: "SLSN-I",
    14: "99a",
    15: "99b",
    16: "99c",
    17: "99d",
}

pb_wavelengths = {
    0: 3685.0,
    1: 4802.0,
    2: 6231.0,
    3: 7542.0,
    4: 8690.0,
    5: 9736.0,
}

pb_colors = {
    0: "#984ea3",  # Purple: https://www.color-hex.com/color/984ea3
    1: "#4daf4a",  # Green: https://www.color-hex.com/color/4daf4a
    2: "#e41a1c",  # Red: https://www.color-hex.com/color/e41a1c
    3: "#377eb8",  # Blue: https://www.color-hex.com/color/377eb8
    4: "#ff7f00",  # Orange: https://www.color-hex.com/color/ff7f00
    5: "#e3c530",  # Yellow: https://www.color-hex.com/color/e3c530
}

config = {
    "num_bins": 500,
    "num_time_bins": 500,
    "max_delta_time": 1000,
    "min_flux": -10000,
    "max_flux": 10000,
    "static_features": ["hostgal_photoz", "hostgal_photoz_err"],
    "num_labels": 18,
    "class_keys": class_keys,
    "class_names": class_names,
    "bands": [0, 1, 2, 3, 4, 5],
    "pb_wavelengths": pb_wavelengths,
    "pb_colors": pb_colors,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sn",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--augment_factor",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--out_suffix",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num_time_bins",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--token_window_size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--format",
        type=str,
        default="tokens",
        choices=["tokens", "gp_tokens", "gp_samples"],
    )
    parser.add_argument(
        "--sample_interval",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


def main(args):
    if not os.path.exists("plasticc"):
        os.makedirs("plasticc")

    for file in files + test_files:
        if not os.path.isfile(os.path.join("plasticc", file)):
            url = "https://zenodo.org/record/2539456/files/%s" % file
            urllib.request.urlretrieve(url, os.path.join("plasticc", file))

    df_train_meta = pd.read_csv("plasticc/plasticc_train_metadata.csv.gz")
    df_train = pd.read_csv("plasticc/plasticc_train_lightcurves.csv.gz")
    print(np.percentile(df_train["flux"], [0.1, 99.9]))

    df_test_meta = pd.read_csv("plasticc/plasticc_test_metadata.csv.gz")
    for file in test_files:
        df_test = pd.read_csv(os.path.join("plasticc", file))
        print(file, np.percentile(df_test["flux"], [0.1, 99.9]))

    config["sn"] = args.sn
    config["augment_factor"] = args.augment_factor
    config["num_time_bins"] = args.num_time_bins
    config["num_bins"] = args.num_bins
    config["token_window_size"] = args.token_window_size
    config["format"] = args.format
    config["sample_interval"] = args.sample_interval

    tokenizer = LCTokenizer(config["min_flux"], config["max_flux"], config["num_bins"], config["max_delta_time"],
                            config["num_time_bins"], bands=config["bands"],
                            transform=np.arcsinh, inverse_transform=np.sinh,
                            min_sn=args.sn, window_size=args.token_window_size)

    config["vocab_size"] = tokenizer.vocab_size

    if args.out_suffix is not None:
        with open("plasticc/dataset_config_%s.json" % args.out_suffix, "w") as f:
            json.dump(config, f)
    else:
        with open("plasticc/dataset_config_%s.json" % args.format, "w") as f:
            json.dump(config, f)

    def load_sequences(df_meta, df, augment_factor=1):
        sequences = []
        if args.format == "gp_tokens":
            for i, row in df_meta.iterrows():
                df_object = df.loc[(df["object_id"] == row["object_id"]), :]
                resampled_df, _ = fit_2d_gp(df_object, config["pb_wavelengths"])
                tokens = tokenizer.encode(resampled_df)
                if len(tokens) > 2:
                    sequences.append({"x": tokens, "class": class_keys[int(row["true_target"])],
                                      "static": row[config["static_features"]],
                                      "object_id": row["object_id"]})
        elif args.format == "tokens":
            token_augs = []
            for i in range(augment_factor):
                token_augs.append(tokenizer.encode(df, augment=i > 0))
            zipped = [df_meta["object_id"], df_meta["true_target"]]
            for static_feature in config["static_features"]:
                if static_feature in df_meta.columns:
                    zipped += [df_meta[static_feature]]
            for row in zip(*zipped):
                for i in range(augment_factor):
                    if len(token_augs[i][id]) >= 2:
                        sequences.append({"x": token_augs[i][id], "class": class_keys[int(row[1])],
                                          "static": list(row[2:]), "object_id": row[0]})
        elif args.format == "gp_samples":
            for i, row in df_meta.iterrows():
                if i % 100:
                    print(i)
                df_object = df.loc[(df["object_id"] == row["object_id"]), :]
                _, (sampled_times, sampled_obs, _, sampled_mask) = fit_2d_gp(df_object, config["pb_wavelengths"],
                                                                             sample_interval=args.sample_interval)
                sequences.append({"sampled_times": sampled_times, "sampled_obs": sampled_obs,
                                  "sampled_mask": sampled_mask,
                                  "class": class_keys[int(row["true_target"])],
                                  "static": row[config["static_features"]],
                                  "object_id": row["object_id"]})
        return sequences

    if args.test_fraction > 0:

        # Make a new representative training split

        df_train_split_meta, df_test_split_meta = train_test_split(df_train_meta, test_size=args.test_fraction,
                                                                   random_state=args.random_state)
        train_sequences = load_sequences(df_train_split_meta, df_train, augment_factor=config["augment_factor"])
        test_sequences = load_sequences(df_test_split_meta, df_train)
        for i, file in enumerate(test_files):
            df_test = pd.read_csv(os.path.join("plasticc", file))
            df_train_split_meta, df_test_split_meta = train_test_split(
                df_test_meta[df_test_meta['object_id'].isin(df_test['object_id'].values)], test_size=args.test_fraction,
                random_state=args.random_state)
            train_sequences += load_sequences(df_train_split_meta, df_test, augment_factor=config["augment_factor"])
            test_sequences += load_sequences(df_test_split_meta, df_test)

    else:

        train_sequences = load_sequences(df_train_meta, df_train, augment_factor=config["augment_factor"])
        test_sequences = []
        for i, file in enumerate(test_files):
            df_test = pd.read_csv(os.path.join("plasticc", file))
            test_sequences += load_sequences(df_test_meta[df_test_meta["object_id"].isin(df_test["object_id"].values)],
                                             df_test)

    if args.out_suffix is not None:
        np.save("plasticc/train_%s.npy" % args.out_suffix, train_sequences)
        np.save("plasticc/test_%s.npy" % args.out_suffix, test_sequences)
    else:
        np.save("plasticc/train_%s.npy" % args.format, train_sequences)
        np.save("plasticc/test_%s.npy" % args.format, test_sequences)

    num_train_sequences = len(train_sequences)
    num_test_sequences = len(test_sequences)

    print('Num train sequences: %s' % num_train_sequences)
    print('Num test sequences: %s' % num_test_sequences)

    if args.format in ["tokens", "gp_tokens"]:
        num_train_tokens = len([x for xs in train_sequences for x in xs['x']])
        num_test_tokens = len([x for xs in test_sequences for x in xs['x']])
        print('Num train tokens: %s' % num_train_tokens)
        print('Num test tokens: %s' % num_test_tokens)
        print('Average train tokens: %s' % (num_train_tokens / num_train_sequences))
        print('Average test tokens: %s' % (num_test_tokens / num_test_sequences))
        print('Optimal model parameters (Chinchilla paper): %s' % int(num_train_tokens / 20))


if __name__ == "__main__":
    args = parse_args()
    main(args)
