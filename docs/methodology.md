# Methodology

## Protocol
* Open-set: unseen classes only in test. Train/val/test = stratified over BENIGN + known attacks.
* Thresholds/calibration from benign validation only. Report operating point (default 5% FPR) *and*
  threshold-free AUROC separately for known and unseen classes.
* All experiments repeated over seeds; report mean ± std. Use paired comparisons across seeds when claiming differences.
* Feature windows (60 s) and the novelty baseline (first 25% of timeline, benign) are label-free.

## Metrics
AUROC / recall@FPR per class group, closed-set macro-F1 for known classes, `unseen_labelled_unknown`
(flagged *and* reported as unseen), per-class forgetting and backward transfer (`forgetting_summary`).

## Threats to validity
* Dataset artefacts (CICIDS2017 labelling, flow-timeout effects) can inflate scores; use CSE-CIC-IDS2018 and UNSW-NB15 as replication.
* Random (non-temporal) splits leak concept drift into optimistic estimates; add a temporal split for deployment claims.
* `dst_unseen` is a strong feature for botnet/C2 traffic; run the ablation `features.use_novelty: false`.
* Transductive graph construction (see README). Add a streaming-window evaluation before claiming online performance.
* Synthetic data results are for testing the pipeline, not for publication.
