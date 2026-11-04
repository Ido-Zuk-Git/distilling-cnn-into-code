# Distilling a neural network into code

A coding agent gets a small CNN trained on 505 Imagenette images (numpy weights and activations) and must hand-write a pure-program classifier, `classify.py`, with no statistical learning, no numbers copied from the network, no fitting to it and no image viewer. The program is scored once on 2,000 hidden images.

Write-up: [Distilling a neural network into code](https://ido-zuk-git.github.io/2026/11/04/distilling-a-cnn-into-code.html).

Result of the published run (Codex `gpt-6.1-sol`, high effort, 25 versions, 44 minutes): **40.80%** on the hidden test, teacher **68.90%**. Full numbers in `results/run/hidden_test_v6_sol.json` and `baseline/results_505/summary.md`.

## What is here

```
rules/          AGENTS.md and BRIEF.md templates the agent receives
scripts/        run_driver.sh (session loop), score.py (scorer), subagent_daemon.py, budget_monitor.py
workspace/      what the agent sees: check.py, agent.py, step.py, look.py (disabled), teacher/ (weights + CLI)
data/           splits.json (file lists with hashes), build_splits.py (downloads Imagenette, rebuilds the splits)
baseline/       teacher training script, numpy export, conventional baselines, their results
results/run/    classify_v01.py .. classify_v25.py, NOTES.md, val_passes.json, run_meta.json, event logs, hidden test
```

## Run it

Needs Linux, Python 3.12, git and the [Codex CLI](https://github.com/openai/codex) logged in.

```bash
python3 -m pip install -r requirements.txt
rm -rf results baseline/results_505          # Codex can read the whole repo; do not leave the answer in it
python3 data/build_splits.py --no-hidden     # train (505) and val (495) into workspace/data/
mkdir -p out && nohup python3 scripts/subagent_daemon.py >> out/subagent.stdout 2>&1 &
scripts/run_driver.sh full --no-look --val-budget 25 --stop-at 97
```

The driver runs fresh Codex sessions of at most 40 minutes, handing over the brief and the tail of `NOTES.md`, until the 25 validation passes are spent. If the Codex budget monitor stops it, re-enter the same run later:

```bash
scripts/run_driver.sh continue full --stop-at 97
```

Outputs land in `out/full/`; the deliverables are `workspace/classify.py` and `workspace/NOTES.md`. A short pilot on 30 + 30 images: `scripts/run_driver.sh pilot --no-look`.

Score the hidden test once, after the run:

```bash
python3 data/build_splits.py --hidden-only
python3 scripts/score.py workspace/classify.py --images hidden/test --out out/full/hidden_test.json
```

To retrain the teacher instead of using the shipped `workspace/teacher/cnn_505.npz`, see `baseline/cnn_small.py` and `baseline/export_teacher.py`, then update the target accuracy in `rules/` and `workspace/teacher/README.md`.

## Credits

Prompted by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, MIT CSAIL). The scarce evaluation budget and notebook handoffs follow [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs). Data: [Imagenette](https://github.com/fastai/imagenette).

## License

MIT, see [LICENSE](LICENSE).
