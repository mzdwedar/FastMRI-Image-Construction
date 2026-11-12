# Spec: model-monitoring

Module of [SPEC.md](SPEC.md). Depends on: `observability`, `model-packaging` (baseline stored with the model).

## Objective

Detect when production inputs or outputs drift away from what the model was trained on, and estimate
reconstruction quality without ground truth, using the existing Prometheus/Grafana/Alertmanager stack (no extra
services).

## Requirements

1. **Per-slice statistics in the gateway** (`gateway/stats.py`, numpy, computed on data already in memory;
   overhead budget ≤ 2% of request latency):
   - input: k-space shape (H, W) as a bounded label set (`known` / `other`), log10 k-space energy, pre-normalization
     image max (the `scale`), normalized image mean and std;
   - output: mean, std, fraction of pixels outside [0, 1], non-finite count;
   - **data consistency error** (label-free quality proxy): normalized MSE between the FFT of the reconstruction
     (un-normalized, un-cropped where possible) and the measured k-space at sampled locations. Lower is better;
     values are compared to the baseline, not interpreted absolutely.
2. **Metrics:** `model_input_stat{stat}` and `model_output_stat{stat}` histograms (fixed buckets per stat, defined
   once in code), `model_data_consistency_error` histogram, all labelled `model_version`. Unknown k-space shapes
   increment `model_input_shape_other_total`.
3. **Baseline.** `packaging register` computes the same statistics over a reference set (validation split if
   available, else synthetic set with `baseline_source=synthetic`) and stores `baseline_stats.json` (p05/p25/p50/
   p75/p95 per stat) as a registry artifact. On model load, the gateway exports them as
   `model_baseline_quantile{stat, q, model_version}` gauges.
4. **Drift rules.** Recording rules compute live p50 per stat over 1 h from histograms; alert `ModelInputDrift` /
   `ModelOutputDrift` when live p50 is outside baseline [p05, p95] for 30 m with ≥ 100 slices in the window
   (ticket severity). `ModelOutputNonFinite` pages on any non-finite output. `ModelQualityProxyDegraded` when live
   p50 data-consistency error > baseline p95 for 30 m.
5. **Dashboard** "Model": live vs baseline quantile bands per stat, data consistency error over time, slices by
   model version, unknown-shape rate. Annotations on promotion events (packaging writes a Grafana annotation via
   API on promote/rollback).

## Commands

```
make register CKPT=...       # also writes baseline_stats.json
make drift-demo              # sends a scaled/noisy synthetic volume to trigger ModelInputDrift (for demos and e2e)
```

## Testing strategy

- Unit: each stat on known arrays (golden values); data consistency error is ~0 for a perfect reconstruction of
  fully sampled synthetic k-space and increases monotonically with added noise; stats computation time ≤ 2% of a
  benchmark request (measured on a 35-slice volume).
- `promtool test rules`: drift alert fires when synthetic histograms shift outside baseline bands and stays quiet
  with < 100 slices.
- E2E: `make drift-demo` makes `ModelInputDrift` reach `alert-sink` (with shortened `for:` via test override).

## Boundaries

- Always: compute stats on data already in memory; bounded labels only.
- Ask first: storing any per-request data (images, k-space) for offline analysis; adding a drift service
  (Evidently etc.).
- Never: block or alter a response because of drift; log image content.

## Success criteria

- [ ] Every registered version has `baseline_stats.json`; gateway exports baseline gauges for the served version.
- [ ] Drift and non-finite alerts covered by rule tests and the drift demo e2e.
- [ ] Gateway p95 latency regression from stats ≤ 2% in `make bench`.

## Open questions

1. Data consistency needs the mask: with fully sampled input (no `acceleration`), every location is "sampled" and
   the metric measures fidelity to the input. Acceptable as a proxy? Proposal: yes, labelled clearly in the
   dashboard.
