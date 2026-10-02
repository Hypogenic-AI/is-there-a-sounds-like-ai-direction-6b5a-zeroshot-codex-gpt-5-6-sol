# Is there a “sounds like AI” direction?

This repository tests whether a residual-stream direction that separates human from AI text is also a specific, quality-preserving causal control for generation.

## What was run

- Qwen2.5-3B base and instruct models.
- 240 matched HC3 human/ChatGPT pairs for estimation; 60 held-out pairs for readout.
- Layer-30 steering at −2, −1, 0, +1, and +2 projection SD on 36 questions.
- Equal-norm random steering, a human-style prompt, and observable confound controls.
- Two public RoBERTa detectors, DistilGPT-2 perplexity, cross-encoder relevance, length, and repetition.

## Main finding

Held-out readout AUROC is 1.000 instruct and 0.998 base; layer-30 directions have cosine 0.974. But causal specificity is not established. On the older detector, ±2 target steering changes mean machine probability by −0.125/+0.127, while random steering changes it by −0.131/+0.093. The HC3-detector reduction at −2 (−0.441) comes with a +8.82 perplexity penalty. Linear decodability does not establish a distinct, quality-preserving “AI-ness” axis.

The paper is [paper_draft/main.pdf](paper_draft/main.pdf).

## Reproduce

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
export HF_HOME="$PWD/data/hfhome"
.venv/bin/python src/run_experiment.py --stage readout
.venv/bin/python src/run_experiment.py --stage steer
.venv/bin/python src/run_experiment.py --stage score
.venv/bin/python src/analyze.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper_draft/main.tex
```

The run used an RTX A6000; caches are under ignored `data/`. `src/judge_outputs.py` records an attempted blinded API-judge protocol, but no judge results are used because supplied credentials were rejected. No keys are logged or stored.

## Layout

- `src/`: experiment and analysis code.
- `results/`: raw generations, activations, scores, and summaries.
- `paper_draft/`: LaTeX, bibliography, figure, and compiled paper.
