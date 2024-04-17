# gpt-ts

Install requirements

```
pip install -r requirements.txt
```

Download PLAsTiCC dataset 

```
python prepare_plasticc.py download
```

To upload models to hugging face (with the --push_to_hub option)

```
huggingface-cli login
```

To upload metrics to W&B

```
wandb login
```

## Token based

Prepare PLAsTiCC dataset 

```
python prepare_plasticc.py process --format tokens --token_window_size 1
```

This will produce 2 files in the generated plasticc directory with the following properties 

| File | Num sequences | Num tokens  |
|------|---------------|--------------------|
| train | 7,848         | 494,560            |
| test | 3,492,657     | 117,156,631        |


```
python prepare_plasticc.py process --format tokens --token_window_size 10
```

This will produce 2 files in the generated plasticc directory with the following properties 

| File | Num sequences | Num tokens  |
|------|---------------|-------------|
| train | 7,848         | 914,983     |
| test | 3,492,745     | 246,835,127 |