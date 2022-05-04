# Emotional DailyDialog


Implemented with [OpenNMT](https://opennmt.net) in [PyTorch](https://github.com/pytorch/pytorch) version.
## Quickstart

### Step 1: Download & process the dataset

Download DailyDialog dataset from HuggingFace
```bash
python hf_dataset.py
```

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`
* 
### Step 2： Build Vocabularies.
```bash
python build_vocab.py -config small_ea.yaml -n_sample 60000
```

### Step 3： Training.
```bash
python train.py -config small_train.yaml
```

### Step 4 (optional)： Human-Machine interaction with trained model.
```bash
python test.py $path_to_model_file$
```
