import torch
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model

from model import GPTLanguageModel, AutoRegressiveRNN

from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
import wandb
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="plasticc",
        choices=["plasticc"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt",
        choices=["gpt", "rnn"],
    )
    parser.add_argument(
        "--model_weights",
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
        choices=["pretrain_lm", "pretrain_class", "finetune_lm", "finetune_class", "finetune_last_class"],
    )
    parser.add_argument(
        "--test_fraction",
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
    use_lm_head = args.task in ["pretrain_lm", "finetune_lm"]

    batch_size = args.batch_size
    epochs = args.num_epochs
    eval_iters = args.eval_iters
    eval_interval = args.eval_interval
    min_iters_save = args.min_iters_save
    learning_rate = args.learning_rate

    wandb_log = args.logger == 'wandb'
    if wandb_log and args.wandb_api_key is not None:
        wandb.login(key=args.wandb_api_key)

    with open(os.path.join(dataset, "dataset_config.json")) as f:
        dataset_config = json.load(f)

    vocab_size = dataset_config["vocab_size"]
    n_static = len(dataset_config["static_features"])
    n_labels = dataset_config["num_labels"]

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    model_config = {
        "model_type": model_type,
        "dropout": dropout,
        "n_static": n_static,
        "n_labels": n_labels,
        "n_embd": n_embd,
        "n_layer": n_layer,
    }
    if model_type == "rnn":
        model_config["n_hidden"] = n_hidden
    else:
        model_config["n_positions"] = n_positions
        model_config["n_head"] = n_head
        model_config["position_embedding"] = position_embedding

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f)
    else:
        with open(os.path.join(dataset, "model_config.json"), "w") as f:
            json.dump(model_config, f)

    training_config = {
        "task": args.task,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "test_fraction": args.test_fraction,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_type == 'gpt':
        model = GPTLanguageModel(vocab_size, n_head, n_embd, n_positions, n_layer, dropout=dropout,
                                 n_static=n_static, n_labels=n_labels,
                                 position_embedding=position_embedding, use_lm_head=use_lm_head)
    elif model_type == 'rnn':
        model = AutoRegressiveRNN(vocab_size, n_embd, n_hidden, n_static=n_static, n_labels=n_labels,
                                  num_layers=n_layer, dropout=dropout, use_lm_head=use_lm_head)

    model = model.to(device)
    print(model)

    if args.model_weights is not None:
        model.load_state_dict(torch.load(args.model_weights, map_location=torch.device(device)))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Model parameters: {params:,}')

    if args.task in ["finetune_lm", "finetune_class", "finetune_last_class"]:
        target_modules = []
        for n, m in model.named_modules():
            if ('query' in n or 'value' in n) and type(m) == torch.nn.modules.linear.Linear:
                target_modules.append(n)
        if args.task == "finetune_lm":
            modules_to_save = ["lm_head"]
        else:
            modules_to_save = ["class_head"]
        config = LoraConfig(
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    if args.test_fraction > 0:
        train_sequences = []
        test_sequences = []
        train_split, test_split = train_test_split(list(np.load(os.path.join(dataset, 'train.npy'), allow_pickle=True)),
                                                   test_size=args.test_fraction, random_state=42)
        train_sequences += train_split
        test_sequences += test_split
        train_split, test_split = train_test_split(list(np.load(os.path.join(dataset, 'test.npy'), allow_pickle=True)),
                                                   test_size=args.test_fraction, random_state=42)
        train_sequences += train_split
        test_sequences += test_split
    else:
        train_sequences = list(np.load(os.path.join(dataset, 'train.npy'), allow_pickle=True))
        test_sequences = list(np.load(os.path.join(dataset, 'test.npy'), allow_pickle=True))

    num_train_sequences = len(train_sequences)
    num_test_sequences = len(test_sequences)
    num_train_tokens = len([x for xs in train_sequences for x in xs['x']])
    num_test_tokens = len([x for xs in test_sequences for x in xs['x']])

    print('Num train sequences: %s' % num_train_sequences)
    print('Num test sequences: %s' % num_test_sequences)
    print('Num train tokens: %s' % num_train_tokens)
    print('Num test tokens: %s' % num_test_tokens)
    print('Average train tokens: %s' % (num_train_tokens / num_train_sequences))
    print('Average test tokens: %s' % (num_test_tokens / num_test_sequences))
    print('Optimal model parameters (Chinchilla paper): %s' % int(num_train_tokens / 20))

    training_config['num_train'] = num_train_sequences
    training_config['num_test'] = num_test_sequences

    training_ids = np.array([x['object_id'] for x in train_sequences], dtype=np.int64)
    test_ids = np.array([x['object_id'] for x in test_sequences], dtype=np.int64)
    if args.output_dir is not None:
        np.save(os.path.join(args.output_dir, "training_ids.npy"), training_ids)
        np.save(os.path.join(args.output_dir, "test_ids.npy"), test_ids)
    else:
        np.save(os.path.join(dataset, "training_ids.npy"), training_ids)
        np.save(os.path.join(dataset, "test_ids.npy"), test_ids)

    def get_batch(split, batch_size=32, shift=True, repeat_class=True):
        # generate a small batch of data of inputs x and targets y
        # Hugging face models expect non shifted labels
        if split == 'train':
            data = train_sequences
        elif split == 'test':
            data = test_sequences
        elif split == 'val':
            data = test_sequences
        else:
            raise ValueError
        x, y, static = [], [], []
        for ix in np.random.randint(0, len(data), (batch_size,)):
            sequence = data[ix]['x']
            if use_lm_head:
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
        if use_lm_head or repeat_class:
            y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        else:
            y_padded = torch.tensor(y, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, x_padded.size(1)), dtype=torch.float32)
        for i, seq in enumerate(x):
            attention_mask[i, :len(seq)] = 1
            if use_lm_head or repeat_class:
                # -100 gets ignored by torch cross entropy loss
                y_padded[i, len(seq):] = -100
        static = torch.tensor(static, dtype=torch.float32)
        return x_padded.to(device), y_padded.to(device), attention_mask.to(device), static.to(device)

    @torch.no_grad()
    def estimate_loss(eval_iters):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            last_losses = torch.zeros(eval_iters)
            correct = 0
            total = 0
            last_correct = 0
            last_total = 0
            for k in range(eval_iters):
                X, Y, attention_mask, static = get_batch(split)
                output = model(X, labels=Y, attention_mask=attention_mask, static=static)
                losses[k] = output.loss.item()
                if output.last_loss is not None:
                    last_losses[k] = output.last_loss.item()
                correct += torch.sum(Y == torch.argmax(output.logits, dim=-1))
                total += torch.sum(attention_mask)
                if output.last_logits is not None and output.last_labels is not None:
                    last_correct += torch.sum(output.last_labels == torch.argmax(output.last_logits, dim=-1))
                last_total += X.shape[0]
            out['%s/loss' % split] = losses.mean()
            out['%s/last_loss' % split] = last_losses.mean()
            out['%s/accuracy' % split] = (correct / total).item()
            out['%s/last_accuracy' % split] = last_correct.item() / last_total
        model.train()
        return out

    estimate_loss(eval_iters)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if wandb_log:
        run = wandb.init(
            project=args.dataset,
            config={**model_config, **dataset_config, **training_config},
        )

    best_loss = np.inf
    best_iter = 0

    max_iters = int(epochs * len(train_sequences) / batch_size)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            metrics = estimate_loss(eval_iters)
            if metrics['val/loss'] < best_loss:
                best_loss = metrics['val/loss']
                best_iter = iter
                if iter > min_iters_save:
                    if args.output_dir is not None:
                        torch.save(model.state_dict(), '%s/best_weights.pt' % args.output_dir)
                    else:
                        torch.save(model.state_dict(), '%s/best_weights.pt' % dataset)
            if wandb_log:
                wandb.log(metrics)
            print(
                f"step {iter}/{max_iters}: train loss {metrics['train/loss']:.4f}, train accuracy {metrics['train/accuracy']:.4f}, val loss {metrics['val/loss']:.4f}, val accuracy {metrics['val/accuracy']:.4f}")

        # sample a batch of data
        X, Y, attention_mask, static = get_batch('train', batch_size=batch_size)

        # evaluate the loss
        output = model(X, labels=Y, attention_mask=attention_mask, static=static)
        optimizer.zero_grad(set_to_none=True)

        if args.task == "finetune_last_class" and output.last_loss is not None:
            loss = 0.01 * output.loss + output.last_loss
        else:
            loss = output.loss
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    args = parse_args()
    main(args)
