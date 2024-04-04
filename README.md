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

| File | Num sequences | Num tokens (3\sigma) |  Num tokens (2\sigma) |
|------|---------------|----------------------|-----------------------|
| train | 7,848         | 494,560              |  647,577              |
| test | 3,492,890     | 117,156,631          |  170,197,029          |