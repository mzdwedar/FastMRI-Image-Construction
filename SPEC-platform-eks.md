# Spec: platform-eks (approved 2026-11-13; implementation not started)

Module of [SPEC.md](SPEC.md). Depends on: `model-packaging`, `inference-server`, `gateway`, `observability`.
Reuses the EKS platform built in `~/Desktop/k8` (repo `scalable-observed-inference`), milestones M1 to M6.

> **This is a scope change.** SPEC.md decision "Deployment target: Docker Compose on a single host (no
> Kubernetes/cloud in scope)" must be amended before any code lands. Proposal: Compose stays the dev and CI target
> (unit, integration, e2e, bench run there unchanged); EKS becomes the production target.

## Objective

Run the fastmri serving path (gateway → Triton, with the production model from the registry) on the k8 EKS
platform, delivered by GitOps, observed by the same signals as the Compose stack, and sized and hardened using what
the k8 experiments actually measured, not what they predicted.

## What carries over from k8 unchanged

| k8 piece | Where it is | Use here |
|---|---|---|
| Terraform layers: `bootstrap` (persistent: state bucket, ECR) / `network` / `cluster` (ephemeral) | `k8/infra/` | As is; additions below |
| `make up` / `make down` session ritual, `clean-cluster` deletes Ingresses/LBs/PVCs first | `k8/Makefile` | As is |
| Pod Identity (IRSA is blocked by the org SCP) | `k8/infra/cluster/eks.tf` | Workload identity for model download and the LB controller |
| Node SG self-allow rule (module blocked cross-node ports < 1025) | `eks.tf` | As is; Triton gRPC 8001 is above 1025 anyway |
| Argo CD app-of-apps, deploy key outside git, root deleted first on teardown | `k8/gitops/`, `make argocd` | Add fastmri Applications |
| kube-prometheus-stack + metrics-server via Argo, dashboards as ConfigMaps (`grafana_dashboard=1`) | `k8/gitops/apps/`, `k8/dashboards/` | Add Alertmanager, Loki, Tempo, Alloy |
| Chart conventions: probes split startup/readiness/liveness, `maxUnavailable: 0`, preStop sleep, `checksum/config`, replicas omitted when HPA on, non-root + read-only rootfs | `k8/charts/model-server/` | Template for the fastmri chart |
| ECR: immutable SHA tags, scan on push, lifecycle | `k8/infra/bootstrap/ecr.tf` | Two new repos |

## What does not carry over, and why

| k8 choice | Why it doesn't fit fastmri | Replacement |
|---|---|---|
| Weights baked into the image | SPEC.md **Never: bake models into images**; promotion must not need an image build | Model in S3, pulled by an init container (R4) |
| One model-server Deployment | fastmri is two tiers: torch-free gateway + Triton | Two Deployments in one chart |
| CPU HPA at 60% | k8 M6 measured the cliff at CPU 0.60 of request: the HPA fires at the edge, needs ~170 s, pods are dead by then | Standing capacity first; HPA on a queue signal later (R8) |
| No load shedding; liveness killed saturated pods → collapse to 0 ready (k8 M6, twice) | fastmri already has it: gateway semaphore → 503 + `Retry-After`, Triton `max_queue_size` + queue timeout REJECT | Keep, and test that it prevents the k8 collapse (R10) |
| 2 GB Python image, 13.6 s pull | Triton `24.08-py3` is several GB (exact size not measured here); default EKS root disk is 20 GB | Bigger node disk; slim Triton image as follow-up (R2, Q4) |
| SQLite MLflow inside Compose | The cluster is destroyed every session | MLflow stays off-cluster (laptop); cluster reads only S3 (R4) |
| Docker socket for Alloy log discovery | No Docker on EKS nodes; the socket mount was root-equivalent anyway | Alloy DaemonSet reading pod logs via the Kubernetes API/files |

## Target architecture

```
 laptop: MLflow (registry) ── make promote VERSION=n ──┬─▶ s3://<models-bucket>/unet_model/n/{model.onnx,config.pbtxt,model_card.json}
                                                        └─▶ PR: values-prod.yaml  model.version: n
 GitHub ◀── Argo CD watches ─────────────────────────────────────────────────────────────────┐
                                                                                              │
 EKS (2 AZ, private subnets)                                                                  │
   ALB (AWS LB Controller, TLS) ──▶ gateway Deployment (≥2, general nodes) ──gRPC──▶ triton Deployment (≥2, inference nodes)
                                         │ /metrics, OTLP                                 │ init: s3 → emptyDir, sha256 check
                                         ▼                                                ▼ /metrics :8002, OTLP
   kube-prometheus-stack (Prometheus, Alertmanager, Grafana)   Alloy DaemonSet ──▶ Loki, Tempo
```

## Requirements

1. **Spec amendment.** Update SPEC.md decisions table (deployment target, model storage, image registry) and the
   ask-first items this module triggers: S3 artifact store, new services, new runtime dependency (`boto3` or AWS CLI
   in the init image).
2. **Terraform additions (k8 repo; full list in "Platform changes").**
   - `bootstrap`: S3 bucket `…-models` (versioning on, SSE, block public access, no `force_destroy`); ECR repos
     `fastmri/gateway` and `fastmri/triton` (mirror of the pinned Triton image, so pulls are same-region and do not
     depend on nvcr.io).
   - `cluster`: second managed node group `inference`: `m7i-flex.large` (8 GiB; the only Free-Tier-eligible type
     seen in k8 with room for Triton), taint `workload=inference:NoSchedule`, label `role=inference`, **root disk
     50 GiB**, min 2 / max 3, Spot by default as in k8. Pod Identity associations: `triton` SA → read-only on the
     models bucket prefix; `aws-load-balancer-controller` SA → its published policy.
3. **Helm chart** `deploy/helm/fastmri/` (in this repo, so chart and code version together):
   - gateway: Deployment, Service (ClusterIP), PDB `minAvailable: 1`, ServiceMonitor, Ingress (ALB), Secret mount
     for `api_keys` at `/run/secrets/api_keys`, `emptyDir` `/tmp` sized ≥ `MAX_UPLOAD_MB × MAX_INFLIGHT` per pod
     or `MAX_INFLIGHT` lowered to fit (the Compose tmpfs is 512 MB for the same reason).
   - triton: Deployment on `role=inference` with the toleration, **headless** Service (see R7), PDB, ServiceMonitor
     on 8002, flags identical to Compose, `shm` via `emptyDir: {medium: Memory}` on `/dev/shm`.
   - Probes: Triton readiness on **`/v2/models/unet_model/ready`**, not `/v2/health/ready` (inference-server T3:
     24.08 reports server ready for a model that was never found). Liveness on `/v2/health/live` and gateway
     `/healthz` with `timeoutSeconds: 5`, `failureThreshold: 6` (k8 M6: 2 s × 3 killed pods under saturation, and
     an async handler did not fix it). Startup probe budget from the measured model download + load + warmup.
   - CPU request = limit; **ORT intra-op threads rendered from the pod CPU limit**, the same rule as `SERVER_CPUS`
     in Compose (k8 break-it 5: 4 threads on 1 core → p99 2.5 to 4×, 25% throttled at 57% CPU).
   - `values-dev.yaml` (1 gateway, 1 triton, auth may be disabled only with `GATEWAY_ENV=dev`) and
     `values-prod.yaml` (≥2 each, auth on, `model.version` pinned).
4. **Model delivery.** Triton pod init container: download `unet_model/<model.version>/` from S3 to an `emptyDir`,
   verify `model.onnx` sha256 against `model_card.json`, fail (pod never starts) on mismatch. Triton runs
   `--model-control-mode=explicit --load-model=unet_model` against that directory, read-only mount. The model
   version lives in git; Triton's repository load API is not used in-cluster.
5. **Promotion and rollback.** `make promote VERSION=n` keeps every existing gate (preprocess version, PSNR, random
   init) and gains a `--target eks` sync: upload to S3 (idempotent, checksummed), then write `model.version: n` to
   `values-prod.yaml` on a branch and open a PR. Merge → Argo rolls Triton (`maxUnavailable: 0`). Rollback =
   revert the PR; the previous version must still be in S3 (keep the last 3; the bucket lifecycle never deletes the
   version referenced in `values-prod.yaml`).
6. **Mixed versions during a rollout.** For the length of a Triton rollout both versions serve. `X-Model-Version`
   must come from the `model_version` in each Triton response, not from the readiness loop's cache, or responses get
   mislabelled. (Gateway change; needs a gateway spec update.)
7. **gRPC load balancing.** k8 step 10: kube-proxy balances per connection, and one k6 connection hit one pod for
   1296 of 1296 requests. The gateway holds one long-lived gRPC channel per worker, so with a ClusterIP Service each
   worker would pin to one Triton pod. Use a headless Service and a `dns:///` target with `round_robin` load
   balancing in the gRPC channel. **Verify first** that `tritonclient.grpc.aio` accepts channel args for this; if
   not, the fallback is one Triton pod per inference node with the gateway's workers spread across them, and the
   per-pod p99 panel (R9) proves whether it balances.
8. **Capacity and scaling.** Start with fixed replicas sized from a measured knee (R10), not an HPA: k8 showed no
   CPU-driven autoscaler can absorb a 3× spike in ~10 s when time-to-capacity is ~170 s with node room, longer
   without. Gateway HPA (CPU, it is light) and Triton scaling on queue time
   (`nv_inference_queue_duration_us` via prometheus-adapter or KEDA) are follow-ups with their own experiments.
9. **Observability parity.** Alertmanager on (it was off in k8, so nothing paged). Existing
   `monitoring/prometheus/rules/*.yml` delivered as PrometheusRule objects generated from the same files, so
   `promtool test rules` in CI keeps covering them. `build_dashboards.py` output delivered as ConfigMaps. Add a
   **p99 per pod** panel for gateway and Triton (k8: one bad pod of two doubled the service p99 and was invisible in
   the aggregate). Loki, Tempo, Alloy via pinned charts through Argo, retention set small: k8 warned the monitoring
   stack eats 2-vCPU nodes, and model pods went Pending for CPU there.
10. **Measure and break (the deliverable, same loop as k8: predict, build, measure, break, write down).**
    - Triton knee per pod at its real CPU limit, single-slice and 35-slice volumes. Do not extrapolate from the
      Compose baseline (15.5 slices/s on 4 CPUs): the inference node has 2 vCPU on a Flex instance (40% per-vCPU
      baseline, burst limits).
    - Spike to 3× knee: expectation to test is fast 503s and **no** liveness restarts, versus the k8 collapse
      (0 ready pods, 6 restarts). If pods still get killed, the backpressure limits are wrong.
    - Promote and rollback under load: zero failed requests, record time to switch.
    - Kill a Triton pod, drain an inference node, Spot interruption: `/readyz` and alerts behave as in Compose
      success criterion 3.
11. **Edge.** AWS Load Balancer Controller via Argo; ALB Ingress, HTTPS with an ACM certificate, idle timeout above
    the gateway's request deadline. Only the gateway is exposed; Triton has no Ingress and no LoadBalancer Service.
12. **CI (this repo).** `helm lint --strict`, `helm template | kubeconform` against the cluster's Kubernetes
    version (k8 M4: lint passes wrong-shape values and bad indentation; only schema checks catch them), image build
    and scan (push stays manual, see Decisions 2).

## Commands (target)

```
# k8 repo
make up                                   # network + cluster + inference node group + Argo; platform converges from git
make down                                 # unchanged; also removes the ALB via clean-cluster

# this repo
make register CKPT=… SUMMARY=…            # unchanged (local MLflow)
make promote VERSION=n TARGET=eks         # gates → S3 upload → PR bumping values-prod.yaml
make helm-check                           # helm lint --strict + kubeconform
make bench-eks NAME=eks-cpu               # load test through the ALB (or an in-cluster k6 pod, as k8 did)
```

## Testing strategy

- Unit: S3 sync with a stubbed client (checksum mismatch aborts, idempotent re-upload, values file edit is a
  one-line diff); gateway takes `X-Model-Version` from the response.
- Config: `make helm-check` in CI; golden `helm template` output for dev and prod values.
- Cluster (manual, results committed to `results/` and `NOTES`-style tables, since the cluster is ephemeral): the
  R10 experiments.
- Compose e2e stays the PR gate; nothing here runs in CI against a live cluster.

## Boundaries

- Always: model version and image tags change only through git; record measured numbers before `make down`.
- Ask first: GPU node group (not Free Tier, cost changes by an order of magnitude); NAT per AZ; running MLflow
  in-cluster; External Secrets / Secrets Manager instead of a `make`-created Secret.
- Never: expose Triton outside the cluster; `:latest` tags; bake weights into an image; long-lived AWS keys in CI.

## Success criteria

- [ ] `make up` from git alone ends with gateway and Triton Synced + Healthy and `/readyz` 200 through the ALB.
- [ ] A 35-slice synthetic volume round-trips through the ALB with auth; p95 recorded as the EKS baseline.
- [ ] Promote and rollback each complete with zero failed requests under concurrent load; switch time recorded.
- [ ] A 3× spike produces 503s within the queue timeout and zero liveness restarts.
- [ ] One request is visible in Prometheus, Loki and Tempo, as Compose criterion 4.
- [ ] Killing Triton fires `TritonDown` in Alertmanager within 2 min.
- [ ] Measured knee per Triton pod and the resulting replica count written down with the tradeoffs.

## Platform changes (k8 repo, `scalable-observed-inference`)

The platform is reused, not copied, but it has to change. Everything below lands in the k8 repo; nothing in this
list touches fastmri code.

| Area | File(s) | Change | Why |
|---|---|---|---|
| Persistent layer | `infra/bootstrap/` | S3 models bucket (versioned, SSE, public access blocked, no `force_destroy`); ECR `fastmri/gateway`, `fastmri/triton` (immutable, scan, lifecycle) | Models and images must survive `make down` |
| Cluster | `infra/cluster/eks.tf`, `variables.tf` | Node group `inference`: `m7i-flex.large`, 50 GiB disk, taint `workload=inference:NoSchedule`, label `role=inference`, min 2 / max 3 | Triton needs 8 GiB and a big image; keep it off the general nodes running Argo and monitoring |
| Cluster | `infra/cluster/` | Pod Identity associations: `fastmri/triton` → read models bucket; `kube-system/aws-load-balancer-controller` → LB controller policy | Workload identity (IRSA is blocked) |
| Cluster | `infra/cluster/` | Tag public subnets `kubernetes.io/role/elb=1` (check `infra/network` first) | ALB subnet discovery |
| Bootstrap script | `Makefile` | `argocd-repo-secret` handles a second repo (fastmri deploy key, own path under `~/.config/k8-learning/`); `make api-keys-secret` creates the gateway `api_keys` Secret from a hash file outside git | Second Argo source; secrets stay out of git |
| GitOps | `gitops/apps/` | New Applications: `aws-load-balancer-controller`, `loki`, `tempo`, `alloy`, `fastmri` (points at the fastmri repo, `deploy/helm/fastmri`, `values-prod.yaml`); all charts pinned | Edge, logs, traces, the app |
| GitOps | kube-prometheus-stack values | Alertmanager on, with the fastmri receivers (alert-sink equivalent + optional Slack); `ruleSelector`/`serviceMonitorSelector` accept objects from the `fastmri` namespace | Alerts must page; k8 had it off |
| GitOps | `gitops/apps/model-server.yaml` | Remove, or keep scaled to 0 behind a flag | It competes for the 2-vCPU general nodes |
| Teardown | `Makefile` `clean-cluster` | No change expected (already deletes Ingresses first, which removes the ALB); verify that the ALB and its security groups are gone before `terraform destroy` | Orphaned ALBs block VPC deletion and keep billing |
| Cost note | `README.md` | Add the ALB (~hourly + LCU) and 2 inference nodes to the per-hour estimate | Destroy-every-session budget changes |

Order: bootstrap → cluster/node group → Makefile secrets → LB controller → observability apps → `fastmri` app.
The fastmri chart (R3) must exist before the `fastmri` Application is merged, or Argo shows it Degraded.

## Decisions (approved 2026-11-13)

1. **Repo split:** platform (Terraform, Argo, monitoring stack) stays in the k8 repo; the fastmri chart lives in
   this repo; a k8 `gitops/apps/fastmri.yaml` Application points at it, with a second deploy key.
2. **CI image push:** push from the laptop, as k8 did (GitHub OIDC to AWS is blocked by the org SCP denying
   `iam:CreateOpenIDConnectProvider`). CI builds and scans but does not push to ECR. Revisit if the SCP changes.
3. **Hardware:** stay on Free Tier, 2-vCPU nodes, about 1.5 CPU per Triton pod; capacity is whatever R10 measures.
   GPU or larger nodes stay ask-first.
4. **Triton image:** mirror `24.08-py3` into ECR as is; measure the in-cluster pull; build a slim ONNX-Runtime-only
   image only if the number justifies it.

## Open questions

1. **Pod Identity in Triton's own S3 client.** Triton can read `s3://` repositories directly, but whether its AWS
   SDK picks up Pod Identity credentials is unverified. The init container (R4) avoids the question and keeps the
   checksum check; revisit only if init time matters.
