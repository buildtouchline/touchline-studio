# Candidate training status

`runner.py` supplies a bounded candidate-training loop: pretrained initialization, two-class head, frozen backbone, batch one FP32, loss logs and final export. `evaluate.py` adds COCO detector evaluation; the notebook compares identical validation IDs. Neither GPU training nor evaluation has run on the user's data. There is no optimizer resume, interruption checkpoint, automatic promotion, custom augmentation or paid deployment. Add/test interruption recovery before long jobs. Time caps are cooperative, not a billing guarantee.

Before training, use `touchline validate-dataset RELEASE.json --training-gate` and visually inspect paired targets. Overfit a tiny reviewed subset as a wiring check, then compare a bounded candidate with the pretrained baseline. Preserve untouched evaluation and dataset/split/recipe hashes.

The current inference adapter accepts a Torch bundle containing `model_state` and `metadata`. Required metadata is `architecture: fasterrcnn_mobilenet_v3_large_320_fpn`, `num_classes: 2`, `class_map: {"0":"background","1":"person"}`. Include additional source/data/split/code/environment/preprocessing provenance in the actual published bundle. This is a starter format, not a claim that a trained checkpoint exists.

Use [the Colab runbook](../../../docs/COLAB_RUNBOOK.md), pin the tested environment, and implement save/reload/resume checks before long runs. Full SAM 3 fine-tuning remains a separately capped benchmark. No training or cloud workload runs on import.
