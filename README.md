# Distilling a CNN into code

A coding agent is given a small trained CNN (the "teacher": numpy weights plus a numpy forward pass, trained on 505
Imagenette images) and must hand-write a pure-program image classifier, `classify.py`, that does what the network
does without any statistical learning: every number in the program must be one the agent chose and can explain, no
number may be copied from the CNN, nothing may be fitted to the teacher, and the agent has no image viewer, so the
network and the raw pixels are its only sources of visual knowledge. The program is graded on 2,000 hidden Imagenette
images. This repository holds the complete harness and the artefacts of one run with a Codex agent (`gpt-6.1-sol`,
high reasoning effort): the hand-written program reaches **40.80%** on the hidden test set, against **68.90%** for the
teacher it was distilled from.

Blog post: <https://example.com/distilling-a-neural-network-into-code> (placeholder)

## The rules

The agent receives the rules in `workspace/AGENTS.md` (rendered from [`rules/AGENTS.md`](rules/AGENTS.md); Codex
loads it at every session start) and the task in [`rules/BRIEF.md`](rules/BRIEF.md). In short:

1. **No statistical learning.** Nothing in `classify.py` may be derived from data by an automated procedure: no
   fitting, regression, clustering, nearest-neighbour or exemplar tables, no statistics computed over training images
   and pasted in, no automated search over thresholds, no pretrained models, no embedded data.
2. **No numbers from the CNN.** The agent may read every weight and activation, but may not copy, round or re-express
   any value from the teacher into the program.
3. **No fitting to the teacher.** Nothing may be tuned to match the teacher's outputs, by script or by hand-search.
4. **No image viewer.** Nothing may describe an image for the agent; it may only compute on pixels and run the
   teacher (on training images only).
5. **Teacher first.** The teacher must be inspected before any image, and its operation should guide the strategy.
6. **Every non-trivial line commented**, every constant explained. `check.py` rejects a program that has numeric lines
   without a nearby comment.
7. **Allowed tools in the program:** `numpy`, `PIL`, `scipy`, `cv2` (hand-designed functionality only; `dnn`, `ml`,
   cascades, file readers etc. are rejected by the scorer) and the standard library. `predict(path)` may read only the
   image it is given. **1 s per image** (a timeout counts as wrong).
8. **Validation is scarce.** `check.py` on `data/val` prints only the overall accuracy, and only 25 times in the whole
   run (the "val passes"). Train gives full per-image and per-class results.
9. **The agent orchestrates itself.** It decides how to use train and val, and may start sub-agents
   (`workspace/agent.py`, same model and rules, fresh context).
10. **Session handoffs.** The run is a loop of fresh Codex sessions of at most 40 minutes. Each new session gets the
    brief, a short "restarted conversation" paragraph and the last 8,000 characters of the agent's notebook
    (`NOTES.md`). The run ends when all val passes are used.

## Repository layout

```
rules/               AGENTS.md and BRIEF.md templates (rendered by scripts/render_rules.py)
scripts/
  run_driver.sh      the driver: renders the rules, runs the session loop, records out/<mode>/
  render_rules.py    template renderer (--no-look keeps the no-viewer blocks, {{VAL_BUDGET}} is substituted)
  score.py           the scorer (byte-identical to workspace/check.py); used for the hidden test
  subagent_daemon.py serves workspace/agent.py requests with fresh `codex exec` sessions
  budget_monitor.py  stops Codex when the 5-hour rate-limit window reaches --stop-at percent (the driver starts it)
workspace/           exactly what the agent sees at the start of a run (plus data/, built below)
  check.py           self-check scorer (rule checks, comment rule, val-pass ledger)
  agent.py           sub-agent client
  step.py            optional batch helper mentioned in the rules
  look.py            image-viewer client; disabled in this run (it refuses to run when "viewer" is false)
  teacher/           the teacher CNN: cnn_505.npz, arch.json, teacher.py (API + CLI), README.md
data/
  splits.json        which Imagenette files form train (505), val (495) and the hidden test (2,000; names only)
  build_splits.py    downloads imagenette2-160 and rebuilds the splits
baseline/            the script that trained the teacher (cnn_small.py), its export to numpy (export_teacher.py),
                     the conventional baselines, and their results on the 505-image fit (results_505/)
results/run/         the published run: classify_v01.py .. classify_v25.py, NOTES.md, val_passes.json,
                     run_meta.json, events_session_1.jsonl, events_session_2.jsonl, hidden_test_v6_sol.json
```

## Reproducing the run

You need Linux (the harness uses `timeout`, `pgrep`, `flock`), Python 3.12, git, and the
[Codex CLI](https://github.com/openai/codex) logged in with access to the model you want to test.

**1. Install.**

    git clone <this repository> distilling-cnn-into-code && cd distilling-cnn-into-code
    python3 -m pip install -r requirements.txt

**2. Remove the published results from the working copy.** Codex's `workspace-write` sandbox blocks the network and
writes outside `workspace/`, but it can read the whole file system. The rules forbid reading outside the workspace,
but do not leave the answer next to it:

    rm -rf results baseline/results_505

(or run from a copy without them).

**3. Build the data, without the hidden test set.**

    python3 data/build_splits.py --no-hidden

This downloads `imagenette2-160.tgz` (about 100 MB) into `data/cache/`, copies the 505 train and 495 val images into
`workspace/data/{train,val}/<class>/` and checks them against the sha256 sums in `data/splits.json`. The split comes
from a seeded shuffle of Imagenette (seed 20261103; the procedure is in `data/splits.json`); five church images were
later moved from val to train, which is why it is 505 / 495 and not 500 / 500. The file lists are authoritative.

**4. The teacher.** Use the shipped one, `workspace/teacher/cnn_505.npz` (3.8 MB), for a faithful replication.
Retraining is possible, but CPU training is not bit-reproducible, so a retrained teacher will differ slightly:

    python3 -m pip install -r baseline/requirements.txt
    python3 data/build_splits.py --hidden-only          # cnn_small.py also scores the hidden test
    python3 baseline/cnn_small.py --train-only --seeds 0 --out-dir out/baselines_505 --save cnn_505.pt
    python3 baseline/export_teacher.py cnn_505.pt       # writes workspace/teacher/cnn_505.npz and arch.json

(If you retrain, delete `hidden/` again before the run and update the 68.90% target in `rules/*.md` and
`workspace/teacher/README.md` to your teacher's hidden score.)

**5. Start the sub-agent daemon** (outside the sandbox, from the repo root; the published run did not use
sub-agents, but they were available):

    mkdir -p out && nohup python3 scripts/subagent_daemon.py >> out/subagent.stdout 2>&1 &

**6. Start the run** with the flags of the published run:

    scripts/run_driver.sh full --no-look --val-budget 25 --stop-at 97

Defaults that the published run used: model `gpt-6.1-sol`, reasoning effort `high`, session wall 40 minutes,
total wall 4 hours, at most 12 sessions. `--stop-at 97` makes the budget monitor stop Codex when 97% of the Codex
5-hour rate-limit window is used. If that happens, wait for the window to reset and re-enter the same run with

    scripts/run_driver.sh continue full --stop-at 97

which is exactly what happened in the published run (session 1 was stopped after 270 s; session 2 was started with
`continue`). The driver makes local git commits before and after the run as a record (`--no-commit` turns them off;
nothing is pushed). Everything is written to `out/full/`: per-session prompts, Codex event streams, last messages,
`val_passes.json` and `run_meta.json`. The agent's deliverables are `workspace/classify.py` and `workspace/NOTES.md`.

`scripts/run_driver.sh pilot --no-look` runs a short pilot on a 30 + 30 image subset (5 val passes) to check the
setup before spending a full budget.

**7. Validation** is measured by the agent itself (`python3 check.py classify.py --images data/val` inside
`workspace/`), at most 25 times; the ledger is `workspace/.val_passes.json`, copied to `out/full/val_passes.json`.

**8. Hidden test: build it only now, and score once.**

    python3 data/build_splits.py --hidden-only
    python3 scripts/score.py workspace/classify.py --images hidden/test \
        --out out/full/hidden_test.json --preds out/full/hidden_test_preds.csv

The hidden set is 200 images per class from Imagenette's official validation folder. This repository ships only the
file list; for a faithful replication the person running the experiment builds it after the agent has finished and
scores the final `classify.py` once. (In the original setup it was kept encrypted during the run.)
`scripts/score.py` copies `classify.py` alone into an empty directory, runs the same AST rule checks as `check.py`,
and prints accuracy, per-class accuracy and the confusion matrix.

## Results of the published run

| | train (505) | val (495) | hidden test (2,000) |
|---|---:|---:|---:|
| **agent program, v25** (`results/run/classify_v25.py`) | 42.77% (216) | 44.85% (222) | **40.80% (816)** |
| teacher CNN (`workspace/teacher/`) | 99.0% | 68.1% | 68.90% |

The program is 36 KB, 258 code lines of which 223 carry comments, and needs 32 ms per image on a CPU (hidden test).
The hidden test was scored once, on v25, the file on disk at the end of the run.

Conventional baselines fitted on the same 505 training images (`baseline/`, details in
[`baseline/results_505/summary.md`](baseline/results_505/summary.md)):

| method | hidden test |
|---|---:|
| small CNN from scratch (= the teacher) | 68.90% |
| HOG + colour histograms + logistic regression | 45.05% |
| HOG + colour histograms + linear SVM | 40.45% |
| 5-NN on 32x32 pixels (cosine) | 27.30% |
| nearest class mean on 32x32 pixels (cosine) | 25.55% |
| 1-NN on 32x32 pixels (cosine) | 22.10% |

Per-class hidden-test accuracy (200 images each):

| class | program v25 | teacher |
|---|---:|---:|
| tench | 29.5 | 80.5 |
| english_springer | 39.0 | 73.0 |
| cassette_player | 63.5 | 74.5 |
| chain_saw | 17.0 | 49.0 |
| church | 26.5 | 71.5 |
| french_horn | 35.0 | 62.0 |
| garbage_truck | 51.5 | 76.5 |
| gas_pump | 45.5 | 64.0 |
| golf_ball | 37.0 | 61.5 |
| parachute | 63.5 | 76.5 |

How the run went (`results/run/run_meta.json`): two sessions, 2,624 s of agent time in total. Session 1 inspected
the teacher, wrote v01 and spent one val pass (24.65%) before the budget monitor stopped it after 270 s. Session 2
(39 minutes) started from the handoff prompt, wrote v02 to v25 and spent the remaining 24 passes, about one per
version. Val accuracy per pass: 24.65, 39.80, 39.19, 39.19, 40.61, 41.01, 40.61, 41.41, 41.01, 35.56, 39.80, 41.01,
41.01, 42.83, 42.83, 43.03, 44.65, 43.84, 44.44, 44.04, 44.24, 44.24, 44.44, 44.85, 44.85. No sub-agents were used.
`results/run/NOTES.md` is the agent's notebook as it stood at the end of the run, with train and val numbers for
every version. The intermediate analysis scripts and reports the notebook mentions are not included.

The event logs are unedited except that the absolute path of the original checkout was replaced by `/path/to/repo`
and the local user name in `ls -l` output by `user`.

## Model and version

The run used the Codex CLI with model `gpt-6.1-sol` at reasoning effort `high` (`codex exec`, `workspace-write`
sandbox, network off), on 2026-11-04. Results depend on the model, its version and the Codex CLI version; a different
model, or the same model name at a later date, can produce a very different program. Pass `--model` and `--effort`
to the driver to test another model. Note also that the budget monitor reads Codex's own rate-limit records
(`~/.codex/sessions/`), so where the stops fall depends on your Codex plan.

## Credits

- Takehi, Teoh, Zhang and Khattab (MIT CSAIL),
  [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), which prompted the
  question of whether a model that can read a network's weights can also write the network as code.
- [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), the source of two protocol ideas used
  here: a scarce evaluation budget (validation as a small number of accuracy-only passes) and handoffs between fresh
  sessions through the agent's own notebook.
- [Imagenette](https://github.com/fastai/imagenette) by fast.ai.

## License

MIT, see [LICENSE](LICENSE).
