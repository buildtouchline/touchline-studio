# Manual verification

Run `python -m unittest discover -s tests -v` and `python tools/check_artifacts.py` from the repository root after installing the data extra. No CI or Actions workflow is included.

44 CPU tests passed on Windows/Python 3.12.14 with generated fixtures. These cover contracts, sampling, tentative tracking, synthetic metric gates, mixed-format ETL and SoccerNet MOT conversion/class/split/origin checks. The synthetic demo completed with 96 decoded/8 selected frames and physical metrics unavailable.

Real datasets, PyAV decoding, Torchvision GPU training/reload, COCO evaluation, SAM 3 and Colab execution have not been tested in the target runtime. The required next gate is a reviewed real release, two-step train/reload, fair baseline comparison and exported candidate. Passing CPU tests does not validate identity or physical measurements.

27 October 2026: notebook setup now activates the source checkout in the current kernel after an editable subprocess install. Two isolated-process regression tests check source import and missing-checkout rejection. All 36 CPU tests pass; target Colab/GPU execution remains unverified.

27 October 2026: a standalone SoccerNet config helper parses the inspected Sequence INI schema, keeps gameID groups together, handles semicolon role/identity fields, excludes ball and leaves review flags false. Three new synthetic-metadata tests pass; all 39 CPU tests pass. The actual 57-sequence Colab dataset still needs user execution/review.

27 October 2026: the documented SoccerNet `other` human class now maps to person, while ball remains ignored and unknown labels still fail. A regression test checks the class mapping and unchanged false review state. All 40 CPU tests pass; the actual Colab dataset run remains pending.

27 October 2026: staging applies the declared frame stride before unused-box geometry checks and reports invalid unused rows. Regression tests cover unused frame 403, selected frame 401 rejection and invalid stride configuration. All 43 CPU tests pass; the real Colab release build remains unverified.

27 October 2026: the SoccerNet config helper now defaults to the inspected export's zero-based coordinates. An end-to-end generated-image regression checks helper → staging → ETL for an x=0 person box. All 44 CPU tests pass. Real-data origin/bounds counts were supplied from the Colab audit; the full real release/GPU execution remains pending.
