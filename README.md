# How Far Is the Training Corpus from Real Infant Input?

Code and derived results for a comparison of the BabyLM training corpora against
SEEDLingS, a corpus of day-long home recordings from 44 infants, together with an
age-of-acquisition curriculum experiment evaluated against that comparison.

## Requirements

```
pip install -r requirements.txt
```

External data, not redistributed here:

| Source | Used for |
|---|---|
| BabyLM Strict-Small and Strict corpora | training data, corpus frequencies |
| `cdi_human.csv` from the BabyLM evaluation package | human AoA norms |
| [seedlings-nouns](https://github.com/BergelsonLab/seedlings-nouns) (CC-BY) | real infant input frequencies |

A GPT-2 tokenizer must be available locally; on clusters without internet access
on compute nodes, save it once with `GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(path)`
and pass that path via `--tokenizer_path`.

## Pipeline

Corpus construction:

```bash
python scripts/01_build_aoa_norms.py --cdi_csv $CDI --out data/aoa_norms.json

python scripts/02_score_corpus.py --data_dir $STRICT_SMALL \
    --aoa_norms data/aoa_norms.json --out_path data/scored.jsonl

python scripts/03_build_curriculum.py --in_path data/scored.jsonl \
    --out_path data/curriculum_train.txt --stats_out data/curriculum_stats.json

python scripts/04_build_baseline.py --in_path data/curriculum_train.txt \
    --out_path data/flat_shuffle_train.txt
```

Training. Three configurations are reported; each is six runs, two conditions by
three seeds. `slurm/train.sbatch` submits one configuration as an array job.

| Configuration | `--max_words` | `--epochs` |
|---|---|---|
| 10M, one pass | 10421796 | 1 |
| 10M, ten epochs | 200000000 | 10 |
| 100M, one pass | 104534122 | 1 |

Analysis:

```bash
# per run
python scripts/06_trajectory_analysis.py --checkpoints_dir $CKPT/$RUN \
    --aoa_norms data/aoa_norms.json --scored_sentences data/scored.jsonl \
    --out_dir results/$RUN

# across seeds, per configuration
python scripts/07_aggregate_seeds.py --results_dir results/ --out results/multiseed.json

python scripts/08_corpus_audit.py --trajectory results/curriculum_seed42/aoa_trajectory.json \
    --seedlings_csv $SEEDLINGS --out results/corpus_audit.json

python scripts/09_subcorpus_audit.py --data_dir $STRICT_SMALL \
    --aoa_norms data/aoa_norms.json --seedlings_csv $SEEDLINGS \
    --trajectory results/curriculum_seed42/aoa_trajectory.json \
    --label 10M --out results/subcorpus_10M.json

python scripts/10_position_control.py --corpus data/curriculum_train.txt \
    --traj results/curriculum_seed4{2,3,4}/aoa_trajectory.json \
    --label "10M, 1 pass" --out results/position_control_10M_1pass.json
```

BLiMP is computed with the official BabyLM evaluation pipeline, not with code in
this repository:

```bash
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name $CKPT/$RUN/final --backend causal --task blimp \
    --data_path evaluation_data/fast_eval/blimp_fast
```

## Results

`results/` holds the derived values behind each table in the paper.

| File | Paper table |
|---|---|
| `corpus_audit.json` | corpus versus real infant input |
| `subcorpus_{10M,100M}.json` | per-source breakdown |
| `multiseed_{10M_1pass,10M_10ep,100M_1pass}.json` | curriculum versus baseline |
| `position_control_*.json` | contribution of presentation order |

The corpus audit and the sub-corpus breakdown involve no model and no training;
they are deterministic given the corpora and the SEEDLingS annotations. Training
runs are seeded, and each reported correlation is a mean over three seeds.

## Licence

Code released under the MIT Licence. The SEEDLingS annotations are CC-BY and
remain subject to their original terms.
