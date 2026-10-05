# Data

`data.source: synthetic` (default) needs no download: `argus generate-data` or the pipeline creates
flows on the fly. The synthetic generator exists for CI, demos and controlled tests. **Report research
numbers on real datasets.**

Real datasets (download manually; check each licence/terms of use and cite as required):

| Dataset | Where | Notes |
|---|---|---|
| CICIDS2017 | Canadian Institute for Cybersecurity (UNB) | Use the *GeneratedLabelledFlows* CSVs - they contain IPs and timestamps. The *MachineLearningCVE* CSVs have no IPs, so topology cannot be recovered and the graph experiments are not meaningful. |
| CSE-CIC-IDS2018 | CIC / AWS Open Data | Same CICFlowMeter schema as CICIDS2017. |
| UNSW-NB15 | UNSW Canberra (Cyber Range Lab) | Supply a CSV with a header row (`srcip,dstip,dsport,proto,dur,spkts,dpkts,sbytes,dbytes,stime,attack_cat`). |

Put CSVs in `data/raw/` and set `data.source: cicids` (or `unsw`) in `configs/default.yaml`.

Label mapping is in `argus.preprocessing.loader.normalize_labels`. All DoS variants collapse to `DDoS`;
`Infiltration`/`Heartbleed` become `Other` and are excluded from the open-set protocol. Document this
mapping, plus the known CICIDS2017 labelling/flow-construction issues reported in the literature, in any write-up.
