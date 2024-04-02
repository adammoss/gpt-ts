# gpt-ts

Install requirements

```
pip install -r requirements.txt
```

Prepare PLAsTiCC dataset 

```
python prepare_plasticc.py
```

This will produce 2 files in the generated plasticc directory with the following properties 

| File      | Num sequences | Num tokens  |
|-----------|---------------|-------------|
| train.npy | 7,848         | 494,560     |
| test.npy  | 3,492,657     | 117,156,631 |