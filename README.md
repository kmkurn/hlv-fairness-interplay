# hlv-fairness-interplay
Code for our paper "On the Interplay between Human Label Variation and Model Fairness"

## Requirements

  1. Linux with CUDA 12.1
  1. Python 3.10+
  1. [pip-tools](https://github.com/jazzband/pip-tools)

Install dependencies with `pip-sync requirements.txt`.

If you have a non-Linux system or a different version of CUDA/Python, you may need to generate the appropriate `requirements.txt` yourself. See the [example command](https://github.com/kmkurn/hlv-fairness-interplay/blob/f9072a5341c49d8c22e98d8949d99a0b002b075f/requirements.txt#L5) for reference.

## Tests

Ensure that everything works by running the tests with `pytest`.

## Data

Download the SBIC dataset from https://maartensap.com/social-bias-frames/. For TAG, please see https://github.com/kmkurn/train-eval-hlv.

## Training

```bash
./run_training.py train with data_dir=data artifacts_dir=artifacts method=ReL
```

The above invocation trains a base RoBERTa on the SBIC data under directory `data` using repeated labelling (ReL) method and saves the training artifacts (incl. the trained model parameters) under directory `artifacts`. Argument `train` is a *command*, while key-value pairs such as `data_dir=data` are *configurations*. The `train` command can be omitted as it is the default command. The full list of commands can be viewed by executing the `help` command without any arguments.

The script also accepts extra configurations as a JSON file, which is useful for hyperparameter tuning with random search (see below).

## Random search

```bash
./generate_configs.py output_dir -n 20
```

The above invocation generates 20 JSON files under directory `output_dir`, each containing batch size and learning rate configurations sampled randomly. Invoke with `-h` or `--help` for detailed usage.

## Evaluation

The `run_training.py` script already performs evaluation after training completes. However, the group-wise aggregation of the fairness score uses geometric rather than arithmetic mean. To use the latter, the `run_eval.py` script can be used.

```bash
./run_eval.py with artifacts_dir=artifacts p_group=1
```

The above invocation evaluates training artifacts under directory `artifacts` using arithmetic mean for the group-wise aggregation when computing the fairness score. To compute the fairness score for each group and class (without aggregation), the `compute_fair_scores.py` script can be used. Invoke this script with `-h` or `--help` option to see its usage.

## MongoDB integration with Sacred

Both `run_training.py` and `run_eval.py` scripts use [Sacred](https://pypi.org/project/sacred/) and have its `MongoObserver` activated by default. Set `SACRED_MONGO_URL` (and optionally `SACRED_DB_NAME`) environment variable(s) to write experiment runs to a MongoDB instance. For example, set `SACRED_MONGO_URL=mongodb://localhost:27017` if the MongoDB instance is listening on port 27017 on the local machine.

## License

Apache License, Version 2.0

## Citation

```
@misc{kurniawan2025a,
  title = {On the {{Interplay}} between {{Human Label Variation}} and {{Model Fairness}}},
  author = {Kurniawan, Kemal and Mistica, Meladel and Baldwin, Timothy and Lau, Jey Han},
  year = 2025,
  eprint = {2510.12036},
  doi = {10.48550/arXiv.2510.12036},
}
```
