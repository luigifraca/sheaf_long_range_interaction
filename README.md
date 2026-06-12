# Sheaf Long-Range Interaction

Reproducible Neural Sheaf Diffusion experiments for synthetic over-squashing
tasks and City-Networks. The upstream `sheaf-mpnn` implementation is kept
unmodified as a pinned Git submodule.

## Setup

```bash
git clone --recurse-submodules <this-repository>
cd sheaf_long_range_interaction
python3.13 -m pip install uv
uv sync --extra dev
source .venv/bin/activate
```

For W&B mirroring, use `uv sync --extra dev --extra wandb`.

The supported NSD grid is:

- maps: General, Orthogonal/Cayley, Diagonal, fixed Identity
- stalk dimensions: 2, 3, 5
- hidden dimensions: 16, 32
- benchmark seeds: 0, 1, 2

That is 24 architectures and 72 seeded runs for every dataset setting.

The optional `analysis` profile adds GCN, GAT, GraphSAGE, MLP, fixed random
orthogonal sheaves, scalar-stalk controls, and constant-total-width
comparisons. It contains 46 representative architectures per dataset setting
and defaults to seed 0. The original 72-run benchmark grid is unchanged.

## Tasks

### Barbell regression

Two configurable cliques are joined by unique bridge edges. Features in the
left and right clique are sampled from `U(-sqrt(3), 0)` and `U(0, sqrt(3))`.
Each node predicts the mean input feature of the opposite clique.

### Graph transfer

The clean protocol includes Ring, CrossedRing, CliquePath, and binary Tree
topologies. Ring variants and CliquePath use exactly `2, 6, 10, 20, 30`
nodes; Tree uses depths `2` through `8`. NSD depth is the actual shortest path
between source and target.

`configs/transfer_legacy.yaml` keeps duplicate edges and uses the test split
for model selection to match the behavior of the reference runner. It should
not be used for primary results.

### City-Networks

Paris and Shanghai are loaded through PyG `CityNetwork` with augmented
37-dimensional features, ten eccentricity classes, and the canonical
10%/10%/80% masks. Benchmark runs use 16 NSD layers.

## Information-Flow Analysis

For a trained model \(Z=f_\theta(X,G)\), the source-to-target block is

\[
J_{v,u}=\frac{\partial z_v}{\partial x_u}\in\mathbb{R}^{C\times F}.
\]

For each focal target \(v\), the analysis extracts its \(L\)-hop induced
subgraph, keeps the local-to-global node map, runs the model in evaluation
mode and FP32, and differentiates the target output with
`torch.func.jacrev`. The stored pair metrics include

\[
\|J_{v,u}\|_1,\qquad \|J_{v,u}\|_F,\qquad
\sigma_{\max}(J_{v,u}),\qquad
\left\|\frac{\partial z_{v,y_v}}{\partial x_u}\right\|_2.
\]

PyTorch's standard ReLU subgradient is used at zero. Synthetic benchmark
subsets use full Jacobian blocks. For Paris and Shanghai, every sampled focal
node also receives a cheaper reverse-mode VJP of the ground-truth logit. The
two scopes are named `full_logit_jacobian` and `ground_truth_logit` and are
never mixed during aggregation. Full matrices and singular spectra are
computed for the deterministic rich subset configured in
`configs/analysis.yaml`.

At graph distance \(h\), the code records both shell total
\(T_h(v)=\sum_{d(u,v)=h} I(v,u)\) and shell mean
\(M_h(v)=T_h(v)/|\{u:d(u,v)=h\}|\). It then reports their focal-node averages,
normalization by the distance-zero value, 95% bootstrap confidence intervals,
the log-scale decay slope, and the influence-weighted radius.

### Pathwise Jacobians

The full Jacobian is decomposed through exact local layer blocks

\[
A_k(a,b)=\frac{\partial h_a^{(k)}}{\partial h_b^{(k-1)}}.
\]

Dynamic programming sums products of these blocks over all monotone geodesic
computational walks. When depth equals source-target distance, this is the
exact contribution of all shortest paths. At greater depth, self or residual
transitions are allowed while detours and backtracking are excluded. A
deterministic canonical shortest-path contribution is also retained.

The resulting comparison contains the full Jacobian, geodesic sum, canonical
path, non-geodesic residual, path cancellation, and geodesic-to-full norm
ratio. Local blocks are differentiated through the complete NSD layer, so
activation derivatives, learned linear maps, residual terms, and
data-dependent restriction-map generators are included. The separately
stored sheaf transport path product is only a structural proxy.

### Sheaf Geometry and Curvature

For every diffusion layer and directed edge, the analysis stores the raw
restriction maps, cross-edge transport, normalized transport, Frobenius and
spectral norms, singular values, condition number, numerical rank, and
effective rank. The scalar effective strength is

\[
\omega_e =
|\alpha|\,\|B_e\|_F/\sqrt{d}.
\]

Original and learned effective Ollivier-Ricci curvature are computed using the
pinned `GraphRicciCurvature` submodule, `OTDSinkhornMix`, and
\(\alpha=0.5\). Learned strengths become metric lengths through
\(\ell_e=(\epsilon+\omega_e)^{-1}\), followed by median normalization. ATD is
not used.

GraphRicciCurvature runs in a separate Python 3.12 environment:

```bash
scripts/setup_curvature_env.sh
export SLRI_CURVATURE_PYTHON="$PWD/.venv-curvature/bin/python"
```

## RunPod GPU Runs

Attach a persistent network volume and point all data and results at it:

```bash
export SLRI_STORAGE_ROOT=/workspace/sheaf-lri-storage
```

Run one task family:

```bash
scripts/run_barbell_gpu.sh --gpu 0
scripts/run_transfer_gpu.sh --gpu 0
scripts/run_cities_gpu.sh --gpu 0
```

Run all task families sequentially on one GPU:

```bash
scripts/run_all_gpu.sh --gpus 0 --sequential
```

Run them concurrently on three GPUs:

```bash
scripts/run_all_gpu.sh --gpus 0,1,2 --parallel
```

All task launchers accept `--profile smoke`, `--seeds`, `--storage-root`,
`--precision`, `--force`, `--dry-run`, and `--wandb`. Completed run IDs are
skipped by default, so rerunning a launcher resumes the grid.

Train the broader analysis model set and analyze completed checkpoints:

```bash
scripts/run_barbell_gpu.sh --gpu 0 --profile analysis --seeds 0
scripts/run_transfer_gpu.sh --gpu 0 --profile analysis --seeds 0
scripts/run_cities_gpu.sh --gpu 0 --profile analysis --seeds 0

scripts/run_analysis_gpu.sh --gpu 0 --profile benchmark \
  --checkpoints initial,best
```

For one run, or for a cheap end-to-end validation:

```bash
slri analyze run <run-id> --checkpoints best
slri analyze run <run-id> --profile smoke --checkpoints best
slri analyze grid --profile smoke --dry-run
```

## Storage

`SLRI_STORAGE_ROOT` is the sole experiment storage boundary:

```text
$SLRI_STORAGE_ROOT/
├── data/
│   ├── raw/city_networks/
│   ├── processed/city_networks/
│   └── generated/<task>/<dataset-hash>/
├── runs/<task>/<run-id>/
│   ├── resolved_config.yaml
│   ├── provenance.json
│   ├── metrics.jsonl
│   ├── summary.json
│   ├── checkpoints/{initial,best}.ckpt
│   ├── analysis/<analysis-id>/
│   │   ├── resolved_analysis.yaml
│   │   ├── provenance.json
│   │   ├── summary.json
│   │   ├── tables/
│   │   ├── matrices/
│   │   └── figures/
│   └── logs/
├── summaries/
└── runs.sqlite
```

Synthetic data is deterministic from its configuration and seed. It is
normally generated in memory; `slri data prepare --config ...` materializes
the exact train/validation/test bundle under its dataset hash. City archives
and project-owned processed tensors are retained separately.

Every run records package versions, project and submodule Git SHAs, host, GPU
environment, full resolved configuration, epoch metrics, best checkpoint, and
final summary. SQLite is an index; the run files are the source of truth.

## Data and Result Retrieval

```bash
# Download cities or materialize synthetic data.
slri data prepare --dataset all
slri data prepare --config configs/barbell.yaml --profile benchmark

# Inspect available data.
slri data list
slri data describe paris

# Inspect and filter runs.
slri runs list --task transfer --variant identity --status completed
slri runs show <run-id>

# Export one run or a comma-separated field=value query.
slri runs export <run-id> --output run.tar.gz
slri runs export 'task=cities,variant=orthogonal' --output cities.tar.gz

# Produce a portable result table.
slri summarize --query 'task=barbell,status=completed' \
  --output barbell-summary.csv
```

Run manifests are written to `summaries/<task>-<profile>-manifest.jsonl`
before execution. They can be used to assign individual run IDs to external
schedulers.

## Analysis Results

Each analysis directory contains the following source-of-truth artifacts:

- `influence_pairs.parquet`: one row per measured source-target pair, with
  distance, measurement scope, Jacobian norms, class-logit gradient, and
  singular diagnostics.
- `influence_hops.parquet`: shell totals, shell means, normalized curves,
  shell sizes, and confidence intervals for each measurement scope.
- `path_jacobians.parquet`: full, geodesic, canonical, and residual norms,
  path counts, cancellation, layerwise gradient flow, and transport-path
  diagnostics.
- `sheaf_edge_geometry.parquet`: restriction and transport spectra,
  anisotropy, effective ranks, condition numbers, and \(\omega_e\) by layer.
- `curvature_edges.parquet`: original curvature, learned effective curvature,
  curvature change, strength, and normalized length per edge and layer.
- `geometry_influence_correlations.parquet`: Spearman correlations and
  distance-controlled regressions predicting measured path influence.
- `layerwise_metrics.parquet`: hidden-state norm, variance, Dirichlet energy,
  optional probes, and task metrics through diffusion depth.
- `synthetic_jacobians.pt` or `sampled_city_jacobians.pt`: retained full
  matrices for reconstructing stored norms, plus deterministic sampled hidden
  embeddings at every layer.
- `figures/*.pdf`: distance decay, full-versus-pathwise Jacobians, curvature,
  bottleneck evolution, and anisotropy spectra.

`summary.json` records row counts, checkpoint, curvature status, influence
radii, decay slopes, and paths to the artifact groups. Comparing `initial`
with `best` analyses exposes training-induced changes at every diffusion
layer. Accuracy or MSE remains available from the parent run summary.

Analyses are indexed in the `analyses` table of `runs.sqlite`; files remain
authoritative. A run export automatically includes all associated analyses.

```bash
# Locate analyses and inspect their exact files.
slri analyses list --query 'task=barbell,variant=general,status=completed'
slri analyses show <analysis-id>
slri analyses files <analysis-id>

# The parent run record lists associated analysis IDs.
slri runs show <run-id>

# Export one analysis, a filtered collection, or a whole run.
slri analyses export <analysis-id> --output analysis.tar.gz
slri analyses export 'task=cities,checkpoint=best' --output cities.tar.gz
slri runs export <run-id> --output run-with-analyses.tar.gz

# Build portable comparison tables.
slri analyze compare --query 'task=transfer,status=completed' \
  --output summaries/transfer
```

## Development

```bash
uv run pytest
uv run ruff check .

# Validate all GPU launchers without requiring CUDA or downloading data.
scripts/run_barbell_gpu.sh --profile smoke --dry-run
scripts/run_transfer_gpu.sh --profile smoke --dry-run
scripts/run_cities_gpu.sh --profile smoke --dry-run
scripts/run_analysis_gpu.sh --profile smoke --dry-run
scripts/run_all_gpu.sh --gpus 0,1,2 --parallel --profile smoke --dry-run
```

See `THIRD_PARTY.md` for upstream licenses and dataset attribution.
