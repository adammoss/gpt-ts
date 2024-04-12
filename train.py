import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model

from models.gpt import GPTModelConfig, GPTModel
from models.rnn import RNNConfig, AutoRegressiveRNN
from models.patchgpt import PatchGPTConfig, PatchGPT
from transformers import GPT2Config, GPT2LMHeadModel
from utils import randint

from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
import wandb
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="plasticc",
        choices=["plasticc"],
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt",
        choices=["gpt", "rnn", "hf_gpt2", "patch"],
    )
    parser.add_argument(
        "--base_model_weights",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--peft_model_weights",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--position_embedding",
        type=str,
        default="relative_key",
        choices=["relative_key", "absolute"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--logger",
        type=str,
        default=None,
        choices=["wandb"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--n_positions",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pretrain_lm",
        choices=["pretrain_lm", "pretrain_mask", "pretrain_class", "finetune_lm", "finetune_class"],
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--min_iters_save",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--last_weight",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--train_file",
        type=str,
        action='append',
        default=[],
    )
    parser.add_argument(
        "--val_file",
        type=str,
        action='append',
        default=[],
    )
    parser.add_argument(
        "--test_file",
        type=str,
        action='append',
        default=[],
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--last_label_only",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--random_mask_ratio",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="arcsinh",
        choices=["arcsinh", "linear"],
    )
    parser.add_argument(
        "--push_to_hub",
        default=False,
        action="store_true",
    )
    return parser.parse_args()


def main(args):
    dataset = args.dataset
    model_type = args.model
    position_embedding = args.position_embedding
    n_positions = args.n_positions
    n_layer = args.n_layer
    n_embd = args.n_embd
    n_head = args.n_head
    dropout = args.dropout
    n_hidden = args.n_hidden
    patch_size = args.patch_size
    random_mask_ratio = args.random_mask_ratio

    batch_size = args.batch_size
    epochs = args.num_epochs
    eval_iters = args.eval_iters
    eval_interval = args.eval_interval
    min_iters_save = args.min_iters_save
    learning_rate = args.learning_rate

    train_files = []
    for file in list(set(args.train_file)):
        train_files += glob.glob(file)
    val_files = []
    for file in list(set(args.val_file)):
        val_files += glob.glob(file)
    test_files = []
    for file in list(set(args.test_file)):
        test_files += glob.glob(file)

    wandb_log = args.logger == 'wandb'
    if wandb_log and args.wandb_api_key is not None:
        wandb.login(key=args.wandb_api_key)

    if args.dataset_config is not None:
        with open(args.dataset_config) as f:
            dataset_config = json.load(f)
    else:
        with open(os.path.join(dataset, "dataset_config.json")) as f:
            dataset_config = json.load(f)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    n_static = len(dataset_config["static_features"])
    n_labels = dataset_config["num_labels"]
    n_channels = len(dataset_config["bands"])

    model_config = {
        "model_type": model_type,
        "dropout": dropout,
        "n_static": n_static,
        "n_labels": n_labels,
        "n_embd": n_embd,
        "n_layer": n_layer,
    }

    if "vocab_size" in dataset_config:
        vocab_size = dataset_config["vocab_size"]
        model_config["vocab_size"] = vocab_size

    if model_type == "rnn":
        model_config["n_hidden"] = n_hidden
    else:
        model_config["n_positions"] = n_positions
        model_config["n_head"] = n_head
        model_config["position_embedding"] = position_embedding
        if model_type == "patch":
            model_config["patch_size"] = patch_size

    training_config = {
        "task": args.task,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "val_fraction": args.val_fraction,
        'train_files': train_files,
        'test_files': test_files,
        'val_files': val_files,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_type == 'gpt':
        if args.task in ["pretrain_lm", "finetune_lm"]:
            head_type = 'lm'
        else:
            head_type = 'classification'
        config = GPTModelConfig(vocab_size=vocab_size, n_head=n_head, n_embd=n_embd,
                                n_positions=n_positions, n_layer=n_layer, dropout=dropout, n_static=n_static,
                                n_labels=n_labels, position_embedding=position_embedding, head_type=head_type)
        model = GPTModel(config=config)
    elif model_type == 'rnn':
        if args.task in ["pretrain_lm", "finetune_lm"]:
            head_type = 'lm'
        else:
            head_type = 'classification'
        config = RNNConfig(vocab_size=vocab_size, n_embd=n_embd, n_hidden=n_hidden, n_layer=n_layer,
                           dropout=dropout, n_static=n_static, n_labels=n_labels, head_type=head_type)
        model = AutoRegressiveRNN(config=config)
    elif model_type == 'patch':
        if args.task in ["pretrain_lm", "finetune_lm"]:
            head_type = 'pretrain_lm'
        elif args.task in ["pretrain_mask"]:
            head_type = 'pretrain_mask'
        else:
            head_type = 'classification'
        config = PatchGPTConfig(patch_size=patch_size, n_channels=n_channels, n_head=n_head, n_embd=n_embd,
                                n_positions=n_positions, n_layer=n_layer, dropout=dropout, n_static=n_static,
                                n_labels=n_labels, position_embedding=position_embedding, head_type=head_type,
                                random_mask_ratio=random_mask_ratio)
        model = PatchGPT(config)
    elif model_type == 'hf_gpt2':
        config = GPT2Config(vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                            n_positions=n_positions)
        model = GPT2LMHeadModel(config)

    model = model.to(device)
    print(model)

    if args.base_model_weights is not None:
        model.load_state_dict(torch.load(args.base_model_weights, map_location=torch.device(device)))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Model parameters: {params:,}')

    if args.task in ["finetune_lm", "finetune_class"]:
        target_modules = []
        for n, m in model.named_modules():
            if ('query' in n or 'value' in n) and type(m) == torch.nn.modules.linear.Linear:
                target_modules.append(n)
        if args.task == "finetune_lm":
            modules_to_save = ["lm_head"]
        else:
            modules_to_save = ["class_head"]
        config = LoraConfig(
            r=args.lora_rank,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        model_config["lora_rank"] = args.lora_rank
        model_config["lora_dropout"] = args.lora_dropout

    if args.peft_model_weights is not None:
        model.load_state_dict(torch.load(args.peft_model_weights, map_location=torch.device(device)))

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f)
    else:
        with open(os.path.join(dataset, "model_config.json"), "w") as f:
            json.dump(model_config, f)

    train_sequences = []
    val_sequences = []
    test_sequences = []
    for train_file in train_files:
        train_sequences += list(np.load(train_file, allow_pickle=True))
    for test_file in test_files:
        test_sequences += list(np.load(test_file, allow_pickle=True))
    if len(val_files) > 0:
        for val_file in val_files:
            val_sequences += list(np.load(val_file, allow_pickle=True))
    else:
        # No test files supplied so create train test split
        train_sequences, val_sequences = train_test_split(train_sequences, test_size=args.val_fraction,
                                                          random_state=args.random_state)

    num_train_sequences = len(train_sequences)
    num_val_sequences = len(val_sequences)
    num_test_sequences = len(test_sequences)
    train_classes = {}
    for train_sequence in train_sequences:
        if train_sequence['class'] in train_classes:
            train_classes[train_sequence['class']]['count'] += 1
        else:
            train_classes[train_sequence['class']] = {'count': 1}
    print('Num train sequences: %s' % num_train_sequences)
    print('Num val sequences: %s' % num_val_sequences)
    print('Num test sequences: %s' % num_test_sequences)
    if model_type == "patch":
        num_train_tokens = 0
        num_val_tokens = 0
        num_test_tokens = 0
        for xs in train_sequences:
            num_train_tokens += int(len(xs['sampled_times']) / patch_size)
        for xs in val_sequences:
            num_val_tokens += int(len(xs['sampled_times']) / patch_size)
        for xs in test_sequences:
            num_test_tokens += int(len(xs['sampled_times']) / patch_size)
    else:
        num_train_tokens = len([x for xs in train_sequences for x in xs['x']])
        num_val_tokens = len([x for xs in val_sequences for x in xs['x']])
        num_test_tokens = len([x for xs in test_sequences for x in xs['x']])
    print('Num train tokens: %s' % num_train_tokens)
    print('Num val tokens: %s' % num_val_tokens)
    print('Num test tokens: %s' % num_test_tokens)
    print('Average train tokens: %s' % int(num_train_tokens / num_train_sequences))
    print('Average val tokens: %s' % int(num_val_tokens / num_val_sequences))
    if num_test_sequences > 0:
        print('Average test tokens: %s' % (num_test_tokens / num_test_sequences))
    if model_type == 'gpt':
        print('Optimal model parameters (Chinchilla paper): %s' % int(num_train_tokens / 20))
    print('\nTraining class counts:')
    for key, value in dataset_config['class_names'].items():
        if int(key) in train_classes:
            print(value, train_classes[int(key)]['count'])
        else:
            print(value, 0)

    training_config['num_train'] = num_train_sequences
    training_config['num_val'] = num_val_sequences
    training_config['num_test'] = num_test_sequences

    training_ids = np.array([x['object_id'] for x in train_sequences], dtype=np.int64)
    val_ids = np.array([x['object_id'] for x in val_sequences], dtype=np.int64)
    test_ids = np.array([x['object_id'] for x in test_sequences], dtype=np.int64)
    if args.output_dir is not None:
        np.save(os.path.join(args.output_dir, "training_ids.npy"), training_ids)
        np.save(os.path.join(args.output_dir, "val_ids.npy"), val_ids)
        np.save(os.path.join(args.output_dir, "test_ids.npy"), test_ids)
    else:
        np.save(os.path.join(dataset, "training_ids.npy"), training_ids)
        np.save(os.path.join(dataset, "val_ids.npy"), val_ids)
        np.save(os.path.join(dataset, "test_ids.npy"), test_ids)

    def get_batch(split, batch_size=32, shift='hf' not in model_type, repeat_class=True, self_supervised=True):
        # generate a small batch of data of inputs x and targets y
        # Hugging face models expect non shifted labels
        if split == 'train':
            data = train_sequences
        elif split == 'test':
            data = test_sequences
        elif split == 'val':
            data = val_sequences
        else:
            raise ValueError
        x, y, static = [], [], []
        if args.model == 'patch':
            attention_mask = []
            for ix in np.random.randint(0, len(data), (batch_size,)):
                if args.transform == "arcsinh":
                    x.append(torch.tensor(np.arcsinh(data[ix]['sampled_obs']), dtype=torch.float32).T)
                else:
                    x.append(torch.tensor(data[ix]['sampled_obs'], dtype=torch.float32).T)
                y.append(data[ix]['class'])
                attention_mask.append(torch.tensor(data[ix]['sampled_mask'], dtype=torch.float32).T)
                static.append(data[ix]['static'])
            x_padded = pad_sequence(x, batch_first=True, padding_value=0)
            y_padded = torch.tensor(y, dtype=torch.long)
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        else:
            for ix in np.random.randint(0, len(data), (batch_size,)):
                sequence = data[ix]['x']
                if self_supervised:
                    if shift:
                        x.append(torch.tensor(sequence[0:len(sequence) - 1], dtype=torch.long))
                        y.append(torch.tensor(sequence[1:len(sequence)], dtype=torch.long))
                    else:
                        x.append(torch.tensor(sequence, dtype=torch.long))
                        y.append(torch.tensor(sequence, dtype=torch.long))
                else:
                    if shift:
                        x.append(torch.tensor(sequence[0:len(sequence) - 1], dtype=torch.long))
                    else:
                        x.append(torch.tensor(sequence, dtype=torch.long))
                    if repeat_class:
                        y.append(torch.full((len(sequence) - 1,), data[ix]['class'], dtype=torch.long))
                    else:
                        y.append(data[ix]['class'])
                static.append(data[ix]['static'])
            x_padded = pad_sequence(x, batch_first=True, padding_value=0)
            if self_supervised or repeat_class:
                y_padded = pad_sequence(y, batch_first=True, padding_value=0)
            else:
                y_padded = torch.tensor(y, dtype=torch.long)
            attention_mask = torch.zeros((batch_size, x_padded.size(1)), dtype=torch.float32)
            for i, seq in enumerate(x):
                attention_mask[i, :len(seq)] = 1
                if self_supervised or repeat_class:
                    # -100 gets ignored by torch cross entropy loss
                    y_padded[i, len(seq):] = -100
        static = torch.tensor(np.array(static), dtype=torch.float32)
        return x_padded.to(device), y_padded.to(device), attention_mask.to(device), static.to(device)

    @torch.no_grad()
    def estimate_loss(eval_iters):
        out = {}
        model.eval()
        if num_test_sequences > 0:
            splits = ['train', 'val', 'test']
        else:
            splits = ['train', 'val']
        for split in splits:
            losses = torch.zeros(eval_iters)
            last_losses = torch.zeros(eval_iters)
            correct = 0
            total = 0
            last_correct = 0
            last_total = 0
            for k in range(eval_iters):
                X, Y, attention_mask, static = get_batch(split,
                                                         self_supervised=args.task in ["pretrain_lm", "finetune_lm"])
                if 'hf' in model_type:
                    output = model(X, labels=Y, attention_mask=attention_mask)
                else:
                    output = model(X, labels=Y, attention_mask=attention_mask, static=static)
                losses[k] = output.loss.item()
                if output.logits is not None:
                    if attention_mask is not None:
                        B, T, C = output.logits.shape
                        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
                        last_logits = output.logits[torch.arange(B), seqlens_in_batch - 1]
                        last_labels = Y[torch.arange(B), seqlens_in_batch - 1]
                    else:
                        last_logits = output.logits[torch.arange(B), -1]
                        last_labels = Y[torch.arange(B), -1]
                    last_losses[k] = F.cross_entropy(last_logits, last_labels)
                    correct += torch.sum(Y == torch.argmax(output.logits, dim=-1))
                    total += torch.sum(attention_mask)
                    last_correct += torch.sum(last_labels == torch.argmax(last_logits, dim=-1))
                    last_total += X.shape[0]
            out['%s/loss' % split] = losses.mean()
            out['%s/last_loss' % split] = last_losses.mean()
            if total > 0:
                out['%s/accuracy' % split] = (correct / total).item()
                out['%s/last_accuracy' % split] = last_correct.item() / last_total
        model.train()
        return out

    if device == 'cpu':
        metrics = estimate_loss(2)
        print(metrics)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if wandb_log:
        run = wandb.init(
            project=args.dataset,
            config={**model_config, **dataset_config, **training_config},
        )

    best_loss = np.inf

    max_iters = int(epochs * len(train_sequences) / batch_size)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            metrics = estimate_loss(eval_iters)
            if metrics['val/loss'] < best_loss:
                best_loss = metrics['val/loss']
                if iter > min_iters_save:
                    if args.output_dir is not None:
                        torch.save(model.state_dict(), '%s/best_weights.pt' % args.output_dir)
                    else:
                        torch.save(model.state_dict(), '%s/best_weights.pt' % dataset)
                    if args.push_to_hub:
                        model.push_to_hub('adammoss/%s' % model_type, commit_message=f"Iteration {iter}", blocking=False)
            if wandb_log:
                wandb.log(metrics)
            if 'train/accuracy' in metrics:
                print(
                    f"step {iter}/{max_iters}: train loss {metrics['train/loss']:.4f}, train accuracy {metrics['train/accuracy']:.4f}, val loss {metrics['val/loss']:.4f}, val accuracy {metrics['val/accuracy']:.4f}")
            else:
                print(
                    f"step {iter}/{max_iters}: train loss {metrics['train/loss']:.4f}, val loss {metrics['val/loss']:.4f}")

        # sample a batch of data
        X, Y, attention_mask, static = get_batch('train', batch_size=batch_size,
                                                 self_supervised=args.task in ["pretrain_lm", "finetune_lm"])

        # evaluate the loss
        if 'hf' in model_type:
            output = model(X, labels=Y, attention_mask=attention_mask)
        else:
            output = model(X, labels=Y, attention_mask=attention_mask, static=static)

        optimizer.zero_grad(set_to_none=True)
        if "class" in args.task:
            if attention_mask is not None:
                B, T, C = output.logits.shape
                seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
                last_logits = output.logits[torch.arange(B), seqlens_in_batch - 1]
                last_labels = Y[torch.arange(B), seqlens_in_batch - 1]
                offsets = randint(0, seqlens_in_batch, device=device)
                sliced_logits = output.logits[torch.arange(B), offsets]
                sliced_labels = Y[torch.arange(B), offsets]
            else:
                last_logits = output.logits[torch.arange(B), -1]
                last_labels = Y[torch.arange(B), -1]
            if args.last_label_only:
                loss = F.cross_entropy(last_logits, last_labels, label_smoothing=args.label_smoothing)
            else:
                loss = F.cross_entropy(sliced_logits, sliced_labels, label_smoothing=args.label_smoothing)
        else:
            loss = output.loss
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    args = parse_args()
    main(args)
