import urllib.request
import os.path
import pandas as pd
import numpy as np
import json
import argparse
import time
import glob

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
    "static_features": ["host_photoz", "host_photoz_error"],
    "num_labels": 18,
    "class_keys": class_keys,
    "class_names": class_names,
    "bands": ['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty'],
    "pb_wavelengths": pb_wavelengths,
    "pb_colors": pb_colors,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        type=str,
        default="tokens",
        choices=["tokens", "gp_tokens", "gp_sample"],
    )
    parser.add_argument(
        "--sn",
        type=float,
        default=3.0,
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
        default=1,
    )
    parser.add_argument(
        "--gp_sample_interval",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--gp_max_sequences",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="arcsinh",
        choices=["arcsinh", "linear"],
    )
    parser.add_argument(
        "--file",
        type=str,
    )
    return parser.parse_args()


def load_token_sequences(df_meta, df, tokenizer):
    sequences = []
    tokens = tokenizer.encode(df)
    zipped = [df_meta.index, df_meta["class"]]
    for static_feature in config["static_features"]:
        if static_feature in df_meta.columns:
            zipped += [df_meta[static_feature]]
    for row in zip(*zipped):
        object_id = row[0]
        if len(tokens[object_id]) > 2:
            sequences.append({"x": tokens[object_id], "class": class_keys[int(row[1])],
                              "static": list(row[2:]), "object_id": object_id})
    return sequences


def process(args):
    df_meta = pd.read_hdf(args.file, 'metadata')
    df = pd.read_hdf(args.file, 'observations')
    filename, ext = os.path.splitext(args.file)

    config["format"] = args.format

    if "tokens" in args.format:
        config["sn"] = args.sn
        config["num_time_bins"] = args.num_time_bins
        config["num_bins"] = args.num_bins
        config["min_flux"] = args.min_flux
        config["max_flux"] = args.max_flux
        config["max_delta_time"] = args.max_delta_time
        config["token_window_size"] = args.token_window_size

        if args.transform == "arcsinh":
            transform = np.arcsinh
            inverse_transform = np.sinh
        elif args.transform == "linear":
            transform = lambda x: x
            inverse_transform = lambda x: x
        config["transform"] = args.transform

        tokenizer = LCTokenizer(config["min_flux"], config["max_flux"], config["num_bins"], config["max_delta_time"],
                                config["num_time_bins"], bands=config["bands"],
                                transform=transform, inverse_transform=inverse_transform,
                                min_sn=args.sn, window_size=args.token_window_size, time_column='time',
                                band_column='band', parameter_error_column='flux_error')

        config["vocab_size"] = tokenizer.vocab_size

    elif args.format == "gp_sample":
        config["gp_sample_interval"] = args.gp_sample_interval

    with open("%s_avocado.json" % filename, "w") as f:
        json.dump(config, f)

    if args.format == "tokens":

        sequences = load_token_sequences(df_meta, df, tokenizer)
        np.save("%s_tokens.npy" % filename, sequences)

        num_sequences = len(sequences)

        print("Num sequences: %s" % num_sequences)

        num_tokens = len([x for xs in sequences for x in xs["x"]])
        print("Num tokens: %s" % num_tokens)
        print("Average tokens: %s" % (num_tokens / num_sequences))


if __name__ == "__main__":
    if not os.path.exists("plasticc"):
        os.makedirs("plasticc")
    args = parse_args()
    process(args)
