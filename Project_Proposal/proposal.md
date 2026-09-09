# Eviction-Based Memory Management for 3D Gaussian Splatting SLAM

## Team and Responsibilities
Devon Goshorn <br>
Experience with Robotic Systems and Computer Vision System Development <br>
Infrastructure/Systems Development - Developing an existing 3DGS SLAM pipeline, MonoGS, to include a VRAM limit and an eviction queue that implements a selected pruning heuristic <br>

Phillip Li <br>
Algorithmic Design - Developing custom pruning heuristics to achieve the best metrics for a given VRAM budget <br>

## Problem and Motivation
3DGS SLAM maps grow continuously, causing OOM crashes on hardware with limited memory and power. This causes issues for autonomous, real-time robotics and mobile systems that have these limitations. <br>

## Research Questions and Hypotheses
RQ1: What pruning heuristic, and heuristic application, produces the lowest trajectory error when using a limited memory budget? <br>
H1: A combination of computationally expensive pruning heuristics applied sparsely (e.g. every 100 frames, on loop closure), and computationally cheap heuristics applied more liberally will provide the lowest trajectory error for a given memory budget. <br>

## Related Work


## Proposed System and Approach
We propose a test system built on an existing open source 3DGS SLAM system, MonoGS, that implements the pruning heuristics setup and monitors VRAM utilization.

## Evaluation Plan


## Expected Deliverables


## Timeline and Milestones


## Risks and Mitigations


## Reproducibility Plan


## References
