# Definition of Done

Written on 2026-11-05, before implementation. Boxes are ticked **at the end only**, and only with a number, a
file or a commit link beside them. A ticked box with no evidence counts as unticked.

## A. Process and provenance

- [ ] Branch `dev-test01` holds all work; one PR into `main` from it. — evidence: `pending`
- [ ] Plan and test split manifest committed before any implementation code. — evidence: `pending`
- [ ] Commits are incremental and named for what changed and why (no `fix`, no `wip`). — evidence: `pending`
- [ ] Test split never used for training, tuning or checkpoint selection. — evidence: `pending`
- [ ] Every image in the dataset is one I took with my own camera. — evidence: `pending`
- [ ] Seeds, library versions and training config pinned and committed. — evidence: `pending`

## B. Calibration (item 1)

- [ ] At least 20 ChArUco or checkerboard views captured, board square measured with the calliper. — evidence: `pending`
- [ ] Camera matrix and distortion coefficients saved to a file that every entry point loads. — evidence: `pending`
- [ ] Reprojection error reported, with the image count. — evidence: `pending`
- [ ] Undistortion check shows straight edges staying straight. — evidence: `pending`

## C. Dataset (item 2)

- [ ] 70 or more undistorted images of the object, taken with the calibrated camera. — evidence: `pending`
- [ ] Segmentation masks for every image, human-reviewed, no fabricated annotations. — evidence: `pending`
- [ ] 70/20/10 train/val/test split fixed by the committed file list, no session leakage. — evidence: `pending`
- [ ] Capture log and dataset summary (object, count, split, labelling tool) exist. — evidence: `pending`

## D. Model and test metrics (item 3)

- [ ] Trained segmentation model, not Ultralytics YOLO and not Roboflow; choice justified in writing. — evidence: `pending`
- [ ] Test-split mask IoU reported. — evidence: `pending`
- [ ] Test-split mAP@0.5 reported. — evidence: `pending`
- [ ] Test-split mAP@0.5:0.95 reported. — evidence: `pending`
- [ ] Test-split precision and recall reported, with the definition used. — evidence: `pending`
- [ ] Metric code unit-tested on cases with a known answer. — evidence: `pending`
- [ ] Weights published with a SHA-256 hash in the README. — evidence: `pending`

## E. The command (item 4)

- [ ] One command takes a new image and writes width (mm), height (mm), confidence and a mask overlay. — evidence: `pending`
- [ ] Undistortion, segmentation, reference detection and conversion all happen inside that command. — evidence: `pending`
- [ ] "Object not found" exits cleanly with no number and a clear message. — evidence: `pending`
- [ ] "Reference not found" exits cleanly with no number and a clear message. — evidence: `pending`

## F. Accuracy and the cost of distortion (items 5–6)

- [ ] 10 or more instances measured by hand (calliper) and by the system. — evidence: `pending`
- [ ] Per-instance table with every instance, plus MAE, mean percentage error, median and worst case. — evidence: `pending`
- [ ] Raw outputs saved and the table regenerated from them by script. — evidence: `pending`
- [ ] Same instances measured on raw and on undistorted images, both errors side by side. — evidence: `pending`
- [ ] The difference explained with my own numbers, not theory alone. — evidence: `pending`

## G. Targets and refusal (items 7–9)

- [ ] MO-1 target met, or missed with the reason written down. — evidence: `pending`
- [ ] MO-2 latency measured over five runs, median and spread reported. — evidence: `pending`
- [ ] MO-3 tilt error curve reported with pose-measured angles. — evidence: `pending`
- [ ] Refusal thresholds documented, with the validation data used to choose them. — evidence: `pending`
- [ ] Cost of refusing reported: how many held-out images are declined and what that buys. — evidence: `pending`

## H. Documents and submission

- [ ] `docs/PLAN.md` matches what actually happened, including deviations and their reason. — evidence: `pending`
- [ ] `docs/OBJECTIVES.md` filled from raw outputs, targets met or missed. — evidence: `pending`
- [ ] `docs/DEFINITION_OF_DONE.md` ticked with evidence on every line. — evidence: `pending`
- [ ] `README.md` covers setup, how to run each stage, hardware, dataset summary and weight hash. — evidence: `pending`
- [ ] No dead code, commented-out blocks, stray files or order-dependent notebooks. — evidence: `pending`
- [ ] A 5-minute recording exists, shows a new photo measured on camera, and answers K1–K4. — evidence: `pending`
- [ ] PR link and recording link ready to send. — evidence: `pending`

## I. Not finished (declared, not hidden)

- [ ] Every item I did not reach is listed here with the reason and what it would have cost. — evidence: `pending`
