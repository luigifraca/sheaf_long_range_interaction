# Third-Party Software and Data

## sheaf-mpnn

The unmodified `external/sheaf-mpnn` Git submodule is pinned to commit
`b0fbec5af21bff083d6dc0ea9851bc06211dbe08`.

It is distributed by its authors under Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International. Its source and
license remain in the submodule. This project imports its public Python API but
does not modify or redistribute an adapted copy.

## GraphRicciCurvature

The unmodified `external/GraphRicciCurvature` Git submodule is pinned to commit
`3fcf12b951540d60dd450998c1c5f8ec3752cd46`.

It is distributed under the Apache License 2.0. The project invokes its
`OTDSinkhornMix` Ollivier--Ricci implementation through an isolated Python
3.12 environment because the pinned upstream dependency set is not compatible
with the main Python 3.13 environment.

## On Over-Squashing

The graph-transfer task definitions are clean-room implementations based on:

> Di Giovanni et al., "On Over-Squashing in Message Passing Neural Networks:
> The Impact of Width, Depth, and Topology", ICML 2023.

The reference implementation at `lrnzgiusti/on-oversquashing` is MIT licensed.

## City-Networks

Paris and Shanghai are downloaded through PyTorch Geometric's `CityNetwork`
dataset wrapper. Cite:

> Liang et al., "Towards Quantifying Long-Range Interactions in Graph Machine
> Learning: a Large Graph Dataset and a Measurement", ICLR 2026.

Dataset terms and source attribution remain the responsibility of the original
dataset publishers.
