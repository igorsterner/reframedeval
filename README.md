<div align="center">

<h1 style="text-align: center;">REFRAMED: Towards Realistic Audio Description Generation for Movies</h1>

<div>
  <a href='https://igorsterner.github.io/' target='_blank'><b>Igor Sterner</b></a>&emsp;
  <a href='https://homepages.inf.ed.ac.uk/mlap/' target='_blank'><b>Mirella Lapata</b></a>&emsp;
  <a href='https://homepages.inf.ed.ac.uk/alex/' target='_blank'><b>Alex Lascarides</b></a>&emsp;
  <a href='https://homepages.inf.ed.ac.uk/keller/' target='_blank'><b>Frank Keller</b></a>
</div>

<div>
School of Informatics<br>
University of Edinburgh<br>
United Kingdom
</div>

</div>

<h4>

</h4>

<div align="center">

[![arxiv](https://img.shields.io/badge/arXiv-2601.07765-b31b1b.svg)](https://arxiv.org/abs/2608.09765)
[![Dataset](https://img.shields.io/badge/Dataset-REFRAMED-FFD21E.svg)](https://huggingface.co/datasets/igorsterner/reframed)

</div>

<div align="left">

## Abstract

Audio Description (AD) is a verbal narration of key visual content in videos, enabling access for visually impaired audiences. Unlike standard video captioning, AD is a structured editorial task: descriptions must be inserted into gaps in dialogue and must convey only what is needed to understand the narrative being told. However, existing approaches formulate AD generation in an artificial setting where both the content and timing of descriptions are pre-specified, reducing the task to clip-level captioning. They further rely on noisy transcription and alignment pipelines, and lack the rich parallel data required for modeling narrative context. We introduce a new formulation of AD generation in which models must jointly decide what to describe and when to do it. To support this, we present REFRAMED, a high-quality dataset of 2,023 videos that span 3,302 scenes from 206 movies, with professional AD transcripts (both American and British versions), professional subtitles and aligned screenplays. We also provide a manually curated challenge set that pairs full movies with multiple AD references, together with evaluation protocols that leverage dialogue gaps and multi-reference comparisons. Experiments with state-of-the-art AD systems and multimodal LLMs show that they outperform trivial baselines but fall far short of expert human performance. Our dataset and benchmark establish a new foundation for research on video understanding.


## Installation

#### Coding Environment
```bash
conda create -n reframedeval python=3.13
conda activate reframedeval
```

Clone this repository

```bash
git clone https://github.com/igorsterner/reframedeval
cd reframedeval
```

Then, install the dependencies.

```bash
pip install -r requirements.txt
```

And make sure you have your python path set appropriately.

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```
## Configs

Configs (`configs/*.yaml`) list systems for evaluation. Each config merges with `configs/base.yaml`. See second citation below for details on `realistic`, which now defaults to `true`.

## Data

For the challenge set, the following input data structure is assumed.

```
reference_data/challenge/{subs,scenes,sequences,reference1,reference2}/<movie>.json
generations/challenge/<system>/<movie>.json
```

For the video excerpts in the main REFRAMED dataset:

```
reference_data/videos/<movie>/{subs,reference1,reference2}/<clip>.json
generations/videos/<system>/<movie>/<clip>.json
```

System generations can optionally be in SRT format (`<movie>.srt` / `<clip>.srt`) instead of JSON.

## Preprocessing

```
python src/preprocessing.py --config configs/challenge.yaml
```

Segments raw generations into sentences and description elements, tokenizes, and writes to `preprocessed_data/` (requires Java for the tokenizer).

## Scoring

```
python src/scorer.py --config configs/challenge.yaml
```

Computes scores and writes per-item results for the configured metrics. Optional arguments: `--metrics`, `--metric_subset`.

## Evaluation

```
python src/evaluate.py --config configs/challenge.yaml
```

Prints a latex table of the results. Optional arguments: `--metrics`, `--metric_subset`, `--analysis`, `--significance_testing`.

## Citation

```
@inproceedings{sterner2026reframed,
  title={{REFRAMED}: Towards Realistic Audio Description Generation for Movies},
  author={Igor Sterner and Mirella Lapata and Alex Lascarides and Frank Keller},
  booktitle={Third Conference on Language Modeling},
  year={2026},
  url={https://openreview.net/forum?id=8VyU9qy4Vy}
}
```

For `realistic` scoring, see
```
@misc{sterner-2026-whatwhenhow,
  title={What, When, and How: Audio Description as Constrained Global Optimization}, 
  author={Igor Sterner and Mirella Lapata and Alex Lascarides and Frank Keller},
  year={2026},
  eprint={2609.30121},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2609.30121}, 
}
```