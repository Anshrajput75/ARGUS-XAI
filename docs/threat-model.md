# Threat model

**Assets:** detection integrity, analyst-feedback channel, model artifacts, alert database.

| Threat | Relevance | Mitigation / status |
|---|---|---|
| Evasion (attacker shapes traffic to look benign) | high | multiple heterogeneous detectors; fused `max`; adversarial evaluation is **future work** |
| Poisoning via analyst feedback | medium | feedback is per-flow, attributable (`analyst`), applied via replay buffer; add review/approval + rollback |
| Concept drift | high | PSI/KS drift detector; periodic recalibration of benign thresholds |
| Unsafe deserialisation (joblib/pickle) | medium | load only self-trained artifacts; sign/hash artifacts before production |
| API abuse / DoS | medium | no auth in this reference build: add authentication, rate limiting, TLS before exposing |
| Privacy | medium | flows are metadata only; IPs are personal data under many regimes - minimise retention |
| Explanations leaking decision boundary | low-medium | counterfactuals reveal what would clear an alert; restrict `/explain` to analysts |

Out of scope: encrypted-payload inspection, host telemetry, active response.
