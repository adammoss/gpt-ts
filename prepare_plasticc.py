import urllib.request
import os.path
import pandas as pd
import numpy as np
import json
import argparse
import time
import glob
import multiprocessing

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
    "static_features": ["hostgal_photoz", "hostgal_photoz_err"],
    "num_labels": 18,
    "class_keys": class_keys,
    "class_names": class_names,
    "bands": [0, 1, 2, 3, 4, 5],
    "pb_wavelengths": pb_wavelengths,
    "pb_colors": pb_colors,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        default="tokens",
        choices=["download", "chunk", "process"],
    )
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
        "--max_delta_time",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--min_flux",
        type=int,
        default=-10000,
    )
    parser.add_argument(
        "--max_flux",
        type=int,
        default=10000,
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
        choices=["tokens", "gp_tokens", "gp_sample"],
    )
    parser.add_argument(
        "--sample_interval",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="arcsinh",
        choices=["arcsinh", "linear"],
    )
    parser.add_argument(
        "--small",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--test_file_pattern",
        type=str,
        default="plasticc_test_lightcurves_*.csv.gz",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )
    return parser.parse_args()


def download(full=True):
    files = [
        "plasticc_train_metadata.csv.gz",
        "plasticc_train_lightcurves.csv.gz",
        "plasticc_test_metadata.csv.gz",
        "plasticc_test_lightcurves_01.csv.gz",
    ]
    if full:
        files += [
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
    for file in files:
        if not os.path.isfile(os.path.join("plasticc", file)):
            url = "https://zenodo.org/record/2539456/files/%s" % file
            urllib.request.urlretrieve(url, os.path.join("plasticc", file))


def chunk(args):
    for file in glob.glob(os.path.join("plasticc", args.test_file_pattern)):
        df = pd.read_csv(file)
        object_ids = np.unique(df["object_id"].values)
        num_chunks = int(len(object_ids) / 10000)
        for i, object_ids_chunk in enumerate(np.array_split(object_ids, num_chunks)):
            df_chunk = df[df["object_id"].isin(object_ids_chunk)]
            df_chunk.to_csv(file.split(".csv.gz")[0] + "_chunk%s" % i + ".csv", index=False)


def stats(args):
    df_train = pd.read_csv("plasticc/plasticc_train_lightcurves.csv.gz")
    print("Training percentiles", np.percentile(df_train["flux"], [0.1, 99.9]))
    for file in glob.glob(os.path.join("plasticc", args.test_file_pattern)):
        df_test = pd.read_csv(file)
        print(file, np.percentile(df_test["flux"], [0.1, 99.9]))


def load_token_sequences(df_meta, df, tokenizer, augment_factor=1):
    if not isinstance(df_meta, pd.DataFrame):
        df_meta = pd.read_csv(df_meta)
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    meta = df_meta[df_meta["object_id"].isin(df["object_id"].values)].copy()
    sequences = []
    token_augs = [tokenizer.encode(df, augment=i > 0) for i in range(augment_factor)]
    zipped = [meta["object_id"], meta["true_target"]]
    for static_feature in config["static_features"]:
        if static_feature in meta.columns:
            zipped += [meta[static_feature]]
    for row in zip(*zipped):
        object_id = int(row[0])
        for i in range(augment_factor):
            if len(token_augs[i][object_id]) >= 2:
                sequences.append({"x": token_augs[i][object_id], "class": class_keys[int(row[1])],
                                  "static": list(row[2:]), "object_id": object_id})
    return sequences


def load_gp_token_sequences(df_meta, df, tokenizer):
    if not isinstance(df_meta, pd.DataFrame):
        df_meta = pd.read_csv(df_meta)
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    meta = df_meta[df_meta["object_id"].isin(df["object_id"].values)].copy()
    sequences = []
    for i, row in meta.iterrows():
        df_object = df.loc[(df["object_id"] == row["object_id"]), :]
        resampled_df, _ = fit_2d_gp(df_object, config["pb_wavelengths"])
        tokens = tokenizer.encode(resampled_df)
        if len(tokens) > 2:
            sequences.append({"x": tokens, "class": class_keys[int(row["true_target"])],
                              "static": row[config["static_features"]],
                              "object_id": int(row["object_id"])})
    return sequences


def load_gp_sample_sequences(df_meta, df, sample_interval):
    if not isinstance(df_meta, pd.DataFrame):
        df_meta = pd.read_csv(df_meta)
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    meta = df_meta[df_meta["object_id"].isin(df["object_id"].values)].copy()
    sequences = []
    num = len(meta)
    for i, row in meta.iterrows():
        if i % 100 == 0:
            print("GP %s of %s" % (i, num))
        df_object = df.loc[(df["object_id"] == row["object_id"]), :]
        start_time = time.time()
        _, (sampled_times, sampled_obs, _, sampled_mask), success = fit_2d_gp(df_object,
                                                                              config["pb_wavelengths"],
                                                                              sample_interval=sample_interval)
        sequences.append({"sampled_times": sampled_times, "sampled_obs": sampled_obs,
                          "sampled_mask": sampled_mask,
                          "class": class_keys[int(row["true_target"])],
                          "static": row[config["static_features"]],
                          "object_id": int(row["object_id"]), 'success': success})
    return sequences


def process(args):
    df_train_meta = pd.read_csv("plasticc/plasticc_train_metadata.csv.gz")
    df_train = pd.read_csv("plasticc/plasticc_train_lightcurves.csv.gz")
    df_test_meta = pd.read_csv("plasticc/plasticc_test_metadata.csv.gz")

    config["format"] = args.format
    config["sn"] = args.sn
    config["augment_factor"] = args.augment_factor

    if "tokens" in args.format:
        config["num_time_bins"] = args.num_time_bins
        config["num_bins"] = args.num_bins
        config["min_flux"] = args.min_flux
        config["max_flux"] = args.max_flux
        config["max_delta_time"] = args.max_delta_time
        config["token_window_size"] = args.token_window_size
    elif args.format == "gp_sample":
        config["sample_interval"] = args.sample_interval

    if args.transform == 'arcsinh':
        transform = np.arcsinh
        inverse_transform = np.sinh
    elif args.transform == 'linear':
        transform = lambda x: x
        inverse_transform = lambda x: x

    if "tokens" in args.format:
        tokenizer = LCTokenizer(config["min_flux"], config["max_flux"], config["num_bins"], config["max_delta_time"],
                                config["num_time_bins"], bands=config["bands"],
                                transform=transform, inverse_transform=inverse_transform,
                                min_sn=args.sn, window_size=args.token_window_size)
        config["vocab_size"] = tokenizer.vocab_size

    if args.out_suffix is not None:
        with open("plasticc/dataset_config_%s.json" % args.out_suffix, "w") as f:
            json.dump(config, f)
    else:
        with open("plasticc/dataset_config_%s.json" % args.format, "w") as f:
            json.dump(config, f)

    if args.test_fraction == 0:

        if args.format == "gp_sample":
            #train_sequences = load_gp_sample_sequences(df_train_meta, df_train, args.sample_interval)
            test_files = glob.glob(os.path.join("plasticc", args.test_file_pattern))
            with multiprocessing.Pool(processes=3) as pool:
                test_sequences = pool.starmap(load_gp_sample_sequences,
                                              zip(["plasticc/plasticc_test_metadata.csv.gz"] * len(test_files),
                                                  test_files, [args.sample_interval] * len(test_files)))
                test_sequences = [x for xs in test_sequences for x in xs]
        elif args.format == "tokens":
            train_sequences = load_token_sequences(df_train_meta, df_train, tokenizer,
                                                   augment_factor=config["augment_factor"])
            test_sequences = []
            for i, file in enumerate(glob.glob(os.path.join("plasticc", args.test_file_pattern))):
                df_test = pd.read_csv(file)
                test_sequences += load_token_sequences(df_test_meta, df_test, tokenizer)

    elif args.format == "tokens":

        # Make a new representative training split

        df_train_split_meta, df_test_split_meta = train_test_split(df_train_meta, test_size=args.test_fraction,
                                                                   random_state=args.random_state)
        train_sequences = load_token_sequences(df_train_split_meta, df_train, tokenizer,
                                               augment_factor=config["augment_factor"])
        test_sequences = load_token_sequences(df_test_split_meta, df_train, tokenizer)
        for i, file in enumerate(glob.glob(os.path.join("plasticc", args.test_file_pattern))):
            df_test = pd.read_csv(file)
            df_train_split_meta, df_test_split_meta = train_test_split(
                df_test_meta[df_test_meta['object_id'].isin(df_test['object_id'].values)], test_size=args.test_fraction,
                random_state=args.random_state)
            train_sequences += load_token_sequences(df_train_split_meta, df_test, tokenizer,
                                                    augment_factor=config["augment_factor"])
            test_sequences += load_token_sequences(df_test_split_meta, df_test, tokenizer)

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

    if "tokens" in args.format:
        num_train_tokens = len([x for xs in train_sequences for x in xs['x']])
        num_test_tokens = len([x for xs in test_sequences for x in xs['x']])
        print('Num train tokens: %s' % num_train_tokens)
        print('Num test tokens: %s' % num_test_tokens)
        print('Average train tokens: %s' % (num_train_tokens / num_train_sequences))
        print('Average test tokens: %s' % (num_test_tokens / num_test_sequences))
        print('Optimal model parameters (Chinchilla paper): %s' % int(num_train_tokens / 20))


if __name__ == "__main__":
    if not os.path.exists("plasticc"):
        os.makedirs("plasticc")
    args = parse_args()
    if args.action == "download":
        download(not args.small)
    elif args.action == "chunk":
        chunk(args)
    else:
        process(args)
