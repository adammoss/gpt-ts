import torch
from peft import LoraConfig, get_peft_model

from model import GPTLanguageModel, AutoRegressiveRNN

import os
import numpy as np
import json
import argparse
import random
from tqdm import tqdm

from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        action='append',
        default=[],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
    )
    return parser.parse_args()


def load_model(model_config, model_weights=None, device=None):
    with open(model_config, "r") as f:
        config = json.load(f)
    model_type = config["model_type"]
    if model_type == "gpt":
        model = GPTLanguageModel(**config)
    elif model_type == "rnn":
        model = AutoRegressiveRNN(**config)
    else:
        raise ValueError("Invalid model type")
    print(model)
    if "lora_rank" in config:
        target_modules = []
        for n, m in model.named_modules():
            if ('query' in n or 'value' in n) and type(m) == torch.nn.modules.linear.Linear:
                target_modules.append(n)
        if config['use_lm_head']:
            modules_to_save = ["lm_head"]
        else:
            modules_to_save = ["class_head"]
        config = LoraConfig(
            r=config['lora_rank'],
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            lora_dropout=config['lora_dropout'],
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    model = model.to(device)
    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights, map_location=torch.device(device)))
    return model


def main(args):
    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_config, args.model_weights, device=device)
    test_files = list(set(args.test_file))
    test_sequences = []
    for test_file in test_files:
        test_sequences += list(np.load(test_file, allow_pickle=True))
    model.eval()
    y_true, y_pred = [], []
    random.shuffle(test_sequences)
    if args.max_sequences is not None:
        max_sequences = args.max_sequences
    else:
        max_sequences = len(test_sequences)
    for i in tqdm(range(max_sequences)):
        sequence = test_sequences[i]
        x = torch.tensor(sequence['x'], dtype=torch.long).unsqueeze(0)
        x = x.to(device)
        static = torch.tensor(sequence['static'], dtype=torch.float32).unsqueeze(0)
        static = static.to(device)
        output = model(x, static=static)
        y_true.append(sequence['class'])
        y_pred.append(torch.argmax(output.logits, dim=-1).squeeze(0)[-1].cpu().numpy())
        sequence['logits'] = output.logits.cpu().detach().numpy()
    cm = confusion_matrix(y_true, y_pred)
    if args.output_dir is not None:
        np.save(os.path.join(args.output_dir, 'eval.npy'), test_sequences[:max_sequences])
        np.save(os.path.join(args.output_dir, 'cm.npy'), cm)


if __name__ == "__main__":
    main(parse_args())

