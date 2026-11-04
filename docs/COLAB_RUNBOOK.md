# Colab: dataset to trained person-detector bundle

## Run analysis in Colab — version 1.5

Use [COLAB_ANALYSIS_RUNBOOK.md](COLAB_ANALYSIS_RUNBOOK.md) and notebooks/02_video_analysis.ipynb to run the existing video/profile pipeline in Colab with exported weights. No local Python installation or Modal is required. The training notebook remains unchanged. Drive export verifies generated-file hashes; actual runtime/quality acceptance remains open. 297 guarded offline tests pass.

**First implementation priority.** Open `notebooks/01_dataset_to_model.ipynb` in Google Colab. Run sequentially; edit the marked configuration cells. The notebook uses repository Python modules, so model logic does not fork into a separate notebook implementation. No hosted inference, Modal deployment or paid job is launched.

## 1. Prepare code and runtime

Open the notebook in Colab and leave `REPO_URL` empty to upload the supplied repository starter ZIP through the first cell. It extracts under `/content/touchline-starter`. Alternatively set your repository URL. Use a GPU runtime. Resource availability/session lifetime vary; preflight checks actual Python/Torch/Torchvision/GPU and compiled CPU/CUDA NMS. Preserve a working supplied pair. If incompatible, install a matching pair from the [official PyTorch selector](https://pytorch.org/get-started/locally/), restart and re-run; do not upgrade arbitrary versions separately. [Colab resource FAQ](https://research.google.com/colaboratory/faq.html).

Core installation adds Pillow and pycocotools, not a replacement Torch build. Record `pip freeze`, runtime details and source hashes in the run folder. SAM 3 is a separate optional environment step; do not destabilize the detector environment to fit it into the first run.

Setup also activates the checkout's `src` directory in the running notebook kernel and checks the imported module location. An editable installation in a subprocess alone does not refresh that kernel's startup import hook. This matters for direct imports in the download/reload/export cells, even when CLI subprocesses already work.

## 2. Get actual data

Mount Drive and point `DATA_ROOT` to the extracted image/label archives. Run inventory first. Source repository links are not local training datasets. For SoccerNet, use the optional download cell only after confirming eligibility/access; it explicitly selects `train.zip`, not the whole organization. Inspect disk space first: the inspected training ZIP alone is about 9.58 GB and extraction requires additional space. Pin/record the resolved Hugging Face commit and downloaded file hash. No token is embedded in code; gated resources require the user's account/access.

Safely extract only the intended archive to a fresh data folder. The notebook checks path traversal, symbolic links and expanded size before extraction. Review `seqinfo.ini`, `gameinfo.ini`, frame filenames and `gt/gt.txt`; create the explicit sequence/track-class manifest. These files are not assumed to have identical schema across all SoccerNet datasets. Frame and track IDs do not identify a named player.

Choose either:

- Existing YOLO/COCO/corner/CSV data: copy `examples/etl-config.template.json`, fill class/units/group mappings and paths, then set `CONFIG_PATH`.
- SoccerNet MOT frames: copy `examples/soccernet-config.template.json`, fill track-to-class mapping from actual metadata, frame stride and real match groups/splits, then set `SOCCERNET_CONFIG`. Run `stage-soccernet`; use its generated ETL config.

Templates intentionally fail until review/completeness/eligibility is confirmed. For person detection include players, goalkeepers, referees and other annotated humans as person=1; background=0. Ball needs a later separate detector/class and must not become person. All relevant people must be completely labelled; ignoring visible referees while calling the class generic person creates false background labels.

For the inspected SoccerNet Sequence INI schema (`gameID`, `trackletID_N=role;identity`), draft the actual mappings with `python tools/prepare_soccernet_config.py /content/soccernet-selected --output /content/soccernet-config.json`. This keeps every clip of a game together and creates custom internal holdouts from train.zip. It handles both `goalkeeper` and `goalkeepers`, includes referees/staff and SoccerNet's documented `other` human category, and explicitly ignores ball. Other unknown roles fail. Default stride 25 is about one training image/second for 25 FPS; it does not set the video inference rate. Default pixel origin 0 preserves the inspected SoccerNet export coordinates. Audit bounds/overlays; generic MOT documentation does not establish this export's origin. The config's three review/eligibility flags remain false; set them and a real reviewer only after inspection. `preview_config` in the standalone helper returns raw person-box previews without changing those flags.

Use at least three genuinely independent capture groups with nonempty train/val/test. Preserve official source split membership, hold validation out of training matches, and freeze final test. Source clips from the same game/capture must stay together. The configuration supports explicit `split_assignments`; otherwise ordinary ETL uses deterministic group allocation. Never assert independence just to pass validation.

## 3. Transform and inspect

Run ETL into a fresh `/content` release folder. It checks class decisions, geometry, review flags, duplicate conflicts, capture groups, splits, image hashes and decoded dimensions. It retains image dimensions and converts RGB PNG; it does not resize boxes silently. Inventory alone never clears training readiness.

Run the training gate, then view overlays in the notebook. Inspect all source formats and sample difficult cases, not just the first three images. If any labels are wrong/incomplete, stop, correct and create a new release version. Changing `annotations_reviewed` to true is an attestation, not an automatic review operation. SAM 3 proposals need human correction and a supported reviewed export before inclusion.

## 4. Baseline, bounded training and comparison

Evaluate the official pretrained detector on a bounded validation subset first. Explicit consent enables downloading official weights. Train a tiny candidate for two steps, then load that bundle without downloading any weights and perform inference. This checks targets/loss/checkpoint compatibility. A two-step model is not a useful accuracy result.

For the first real run, increase to a declared cap (for example two epochs, 100 steps, 600 seconds between-step limit) only after the tiny run passes. Record what actually completed; a cap can stop before all epochs. The MobileNet320 starting recipe freezes the backbone and trains RPN/ROI heads with batch one FP32. It trades resolution for a cheap initial runtime; evaluate distant-player recall before deciding if a higher-resolution Torchvision model is needed. [Torchvision fine-tuning contract](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

Evaluate baseline and candidate on the **same validation sample IDs**, using the supplied COCO evaluator. Reports include mAP 0.50–0.95, AP50 and size-bin AP/recall; an undefined bin is null. A bounded subset is explicitly marked and is not full-dataset evaluation. Use all validation images for the selection gate, then inspect field-domain examples. COCO detection scores do not measure identity or physical accuracy. [COCO evaluator source](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

Keep the pretrained rollback if training does not improve the declared criteria. Test is for the selected frozen candidate once, not for choosing thresholds/model variants. The notebook leaves final-test execution behind an explicit switch. No automatic promotion is implemented.

## 5. Export and interruption policy

The export cell copies `candidate.pt`, recipe, result, loss log, evaluations, environment/source record and manifest to durable storage. It also separately copies and revalidates the corresponding release/images under the `dataset-releases` folder; allow sufficient Drive space/time. The small model ZIP excludes those images. The bundle records class map, architecture, dataset manifest hash, seed and versions; it is an unvalidated candidate until evaluation/review passes. Reload preserves the FrozenBatchNorm backbone used during pretrained initialization.

The current training runner saves a final atomic candidate, **not periodic interruption/resume checkpoints**. If Colab disconnects mid-run, logs may survive only where written; restart a small capped run under a new name. Add and verify optimizer/RNG/sampler resume before long runs. Do not assume the cooperative time limit cancels a hung GPU call or bounds provider charges. This is why the initial notebook is deliberately bounded.

## Optional local SAM 3

After the detector path is reproducible, create a separate tested runtime using [Meta's official repository](https://github.com/facebookresearch/sam3). Obtain permitted local weights and record their hash, code revision and actual VRAM. The helper calls local image inference with `load_from_HF=False`; it does not fine-tune SAM 3 or auto-admit proposals. Follow the exact model's requirements rather than promising it fits every Colab GPU. Full SAM 3 training has its own benchmark ticket and budget gate.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing Torch/Torchvision or NMS operator | Install a compatible official pair, restart; capture versions |
| Dataset gate fails | Repair actual class/geometry/group/review data; do not disable the gate |
| Missing YOLO label | Supply a reviewed label, including deliberate empty negative; absence is not a negative |
| Generic CSV/JSON rejected | Inspect semantics and choose/write the right adapter; event statistics are not boxes |
| SoccerNet ball appears as person | Correct sequence-local class decisions; MOT placeholder columns are not class IDs |
| Same match in different splits | Regroup before release; invalidate contaminated evaluation |
| GPU OOM | Stop; record GPU/VRAM; reduce scope/recipe or benchmark another model, not silent retry loops |
| Candidate worse than baseline | Retain baseline; inspect labels/small players/field domain before spending more |
| Output already exists | Use a new run/release path; never overwrite frozen evidence |
| Download finishes, then `No module named touchline` | Keep the ZIP and runtime. Activate the extracted checkout's `src` path in the current kernel; retry only the post-download metadata lines, not the full download |

## Current verification limits

CPU contract/ETL tests run locally. Actual Torchvision/SAM 3 imports, GPU forward/backward, SoccerNet archive conversion, COCO evaluation and Colab execution must be performed in the user's runtime. The notebook contains executable code but has no fabricated executed outputs. A/B should append successful runtime evidence to the referenced ticket completion record.

SoccerNet-Tracking defines the `other` category as medical staff/coaches entering the field; it is mapped to person, not ignored. See section 3 of [the dataset paper](https://arxiv.org/html/2204.06918). Raw label-review/eligibility flags remain false until inspection.

For SoccerNet, frame selection is applied before geometry validation of boxes on unselected frames. Selected frames still require finite positive box extents and then undergo canonical image-bound checks. The staging report records unselected row counts and examples of invalid unused boxes. For stride 25, frame 403 is not selected; a zero-width annotation there does not affect the training image set. A zero-width box on selected frame 401 still fails with sequence/line/frame/track context.

The inspected Colab export audit covered 27,706 selected person boxes: origin 0 gave zero out-of-image boxes; subtracting one introduced 346 invalid edge boxes. Existing configs made with the earlier helper must change each sequence's `bbox_origin` to 0 and rebuild staging and release. Raw annotations are preserved. Recheck the rebuilt overlays before training.
