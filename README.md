# Bilingual Conformer ASR

An attention-based encoder-decoder for English and Italian automatic speech recognition. The encoder combines log-Mel speech features, convolutional subsampling, Transformer self-attention, and optional Conformer convolution blocks. The decoder predicts SentencePiece tokens; multilingual mode prepends an `[ENG]` or `[ITA]` language tag to each transcript.

## Repository contents

- `train.py`, `trainer.py` — training entry point and training loop
- `decode.py` — greedy decoding and WER calculation when references are present
- `models/` — audio frontend, encoder, decoder, attention, and Conformer layers
- `loader.py` — dynamic batching and transcript tokenization
- `specaug.py` — optional SpecAugment implementation
- `bpe.model`, `dict.txt` — SentencePiece tokenizer and vocabulary
- `conf/base.yaml` — example model and training configuration
- `run.sh` — Slurm workflow for training, dev/test decoding, and hypothesis output
- `setup_env.sh`, `pixi.toml`, `pixi.lock` — Delta environment setup
- `Fall_2026___11751___Coding_4___Delta.pdf` — course assignment handout

## Requirements and data

The included setup script targets the CMU Delta cluster and expects the shared Coding 4 dataset under `/work/nvme/bjdh/shared/coding4` by default. It installs the Pixi environment and links the dataset into `dump/`. Override `DATA_ROOT` or `ENV_ROOT` when your cluster paths differ.

```bash
./setup_env.sh
pixi shell
```

The dataset is not included in this repository. The data directory should contain JSON manifests for `train`, `dev`, and `test` in both `monolingual` and `bilingual` forms, along with the referenced audio. Follow the course handout or your dataset provider's instructions to obtain access.

## Train and decode

On Delta, submit one of the supplied Slurm jobs from this directory:

```bash
sbatch run.sh
sbatch --export=ALL,MODE=multilingual run.sh
```

The default job trains the monolingual model. The second command trains in multilingual mode. Each job decodes the dev and test splits; test hypotheses are written to `decoded_hyp_monolingual.txt` or `decoded_hyp_bilingual.txt`.

For direct runs in an activated environment, training and decoding can also be invoked separately:

```bash
pixi run python train.py \
  --tag monolingual --mode monolingual \
  --train_json dump/raw/train_monolingual/data.json \
  --valid_json dump/raw/dev_monolingual/data.json

pixi run python decode.py \
  --exp_dir exp/train_monolingual \
  --ckpt_name CHECKPOINT_NAME \
  --decode_tag test \
  --recog_json dump/raw/test_monolingual/data.json \
  --mode monolingual
```

Use `--mode multilingual` and the corresponding `*_bilingual` manifests for bilingual data. Set `--ckpt_name` to a checkpoint in the experiment's `ckpts/` directory.

## Submission hypotheses

Hypothesis files contain one utterance per line: the utterance ID followed by decoded text. Bilingual output includes its predicted language tag. After generating both files, `prepare_submission.sh` packages them with the Python source files into `submit.zip`.

## Configuration

The model and optimization options are defined in `conf/base.yaml` and can be overridden on the command line. See `python train.py --help` and `python decode.py --help` for the full option lists. Training outputs are written under `exp/`, which is excluded from Git.
