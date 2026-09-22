# DHMD-DBMRE-ACIG

Official implementation of a multimodal sentiment analysis framework
that extends DHMD with Dual-Branch Masked Representation Enhancement
(DBMRE) and Adaptive Cross-modal Interaction Gating (ACIG).

## Overview

Multimodal sentiment analysis (MSA) aims to jointly exploit textual,
acoustic, and visual information for sentiment prediction. However,
multimodal representations may suffer from modality-specific noise,
cross-modal redundancy, and insufficient modeling of informative
cross-modal interactions.

This repository provides the implementation of our proposed framework,
which extends the DHMD framework with two main components:

- **Dual-Branch Masked Representation Enhancement (DBMRE)**  
  A dual-branch masked representation learning module designed to
  enhance modality-specific and modality-invariant representations
  through intra-modal and inter-modal masked learning.

- **Adaptive Cross-modal Interaction Gating (ACIG)**  
  An adaptive gating mechanism that dynamically reweights
  cross-modal interactions, enhancing informative dependencies while
  suppressing noisy and redundant interactions.

The framework further incorporates hierarchical knowledge distillation
and semantic representation learning for multimodal sentiment analysis.

## Framework

The overall framework consists of the following stages:

1. Feature projection and representation decoupling
2. Dual-Branch Masked Representation Enhancement (DBMRE)
3. Hierarchical knowledge distillation
4. Adaptive Cross-modal Interaction Gating (ACIG)
5. Fine-grained semantic matching
6. Multimodal fusion and sentiment prediction

## Repository Structure

```text
DHMD-DBMRE-ACIG/
├── config/
│   └── config.json
├── trains/
│   ├── singleTask/
│   ├── subNets/
│   └── utils/
├── utils/
├── config.py
├── data_loader.py
├── run.py
├── train.py
├── test.py
├── .gitignore
└── README.md
