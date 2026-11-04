# Task

Write `classify.py`: a hand-written program with no statistical learning that defines
`predict(path: str) -> str`, returning one of the 10 Imagenette classes: tench, english_springer,
cassette_player, chain_saw, church, french_horn, garbage_truck, gas_pump, golf_ball, parachute.

Target: 68.90%, the hidden-test accuracy of a small CNN trained from scratch on exactly the 505 training images you have. You are given that CNN (see `teacher/` below). Your goal is to distill the CNN into explainable strategies formulated as code, without any statistical learning, and reach its accuracy that way. Chance is 10%.

# What you have

Your working directory is `workspace/`.

- `data/train/<class>/*.JPEG` -- {{N_TRAIN}} training images in total. JPEG, shortest side
  160 px, variable aspect ratio. Tune your rules on these, in whatever way you decide (AGENTS.md: 'You are the
  orchestrator'); `python3 step.py classify.py` is an optional helper for working in batches.
- `data/val/<class>/*.JPEG` -- {{N_VAL}} validation images in total, same format. Measure only: run
  `python3 check.py classify.py --images data/val` for your estimate of the hidden-test score; it reports the overall
  accuracy and nothing else, and you may run it {{VAL_BUDGET}} times in the whole run (the session continues until
  all {{VAL_BUDGET}} val passes are used). Never view a val
  image, never pass one to the teacher, never read val images from your own scripts. Report train and val
  numbers in NOTES.md for every version you measure.
- `teacher/` -- a small CNN trained from scratch on exactly your 505 training images (68.90% on the hidden
  test), with numpy weights, an architecture description (`teacher/README.md`) and a Python API / CLI
  (`python3 teacher/teacher.py layers | weights | activations | predict | top`; see AGENTS.md). Using it is
  mandatory and comes first: inspect the teacher (its weights, and its activations on training images, with any
  mechanistic interpretability method you want) before you view any image, and let what you learn about how it
  operates guide what your program computes. Train
  images only; it refuses `data/val`. No numbers from it may be copied into `classify.py` and nothing may be
  fitted to its outputs; everything you write must be explainable on its own.
<!-- IF LOOK -->
- `look.py` -- the ONLY way to see an image. You cannot view pixels directly; you can compute
  on them, but to know what an image looks like, use the viewer.
  - `python3 look.py <image> ["question"]` prints a text description from a separate vision
    model. Pass a question to focus it (e.g. "what colours dominate the background?").
  - `python3 look.py --batch <image1> <image2> ... ["question"]` describes several images in
    one call.
  - Train images only, and only after the teacher has been used.
<!-- ENDIF -->
<!-- IF NOLOOK -->
- There is no image viewer in this run. You cannot see pixels directly, but you can compute anything you like on
  them with your own code and print the results, and you can run the teacher on training images; that is how you
  find out what is in an image.
<!-- ENDIF -->
- `check.py` -- self-check. `python3 check.py classify.py --images data/train` (or `--images data/val`)
  runs the same scorer that is used for final scoring, including the rule checks, and prints
  accuracy and any rule violations (on val: the overall accuracy only).
- `agent.py` -- `python3 agent.py "task"` runs a sub-agent (same model and rules, fresh context) on a task in this
  workspace and prints its report. You are the orchestrator: delegate what you want, decide the strategy yourself.

# Rules

See AGENTS.md (loaded automatically). It is binding. The one hard rule is no statistical learning;
every non-trivial line must be commented; the program must be fast and vectorised (1 s per-image timeout).

# Deliverables

- `classify.py` in the workspace root (this is what is scored).
- `NOTES.md` in the workspace root, with your strategy and train/val accuracy for every version you measure.
- As your last message, a 10-line summary of your final approach.
