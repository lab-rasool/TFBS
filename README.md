# Transcription Factor Binding Site (TFBS) Prediction

This project focuses on the prediction of transcription factor binding sites (TFBS) using a mixture of expert (MoE) model and various convolutional neural network (CNN) expert models. The models are evaluated on out-of-distribution (OOD) data to assess their generalization capability.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

Transcription factors (TFs) are proteins that bind to specific DNA sequences, regulating the transcription of genetic information. Accurate prediction of TFBS is crucial for understanding gene regulation and cellular processes. This project employs a mixture of experts (MoE) model to integrate multiple CNN expert models, enhancing the prediction accuracy and robustness on TFBS data.

## Dataset

The project uses ChIP-seq datasets for various transcription factors. The data is divided into training, validation, and test sets, ensuring a balanced representation for model training and evaluation.

## Model Architecture

The model architecture includes multiple CNN expert models, each capturing different characteristics and patterns within the ChIP-seq data. The MoE model integrates the outputs of these experts through a gating network, dynamically weighting their contributions based on the input data.

### CNN Expert Model

- **Convolutional Layer**: Detects motifs within DNA sequences.
- **ReLU Activation**: Introduces non-linearity.
- **Pooling Layer**: Reduces spatial dimensions.
- **Linear Adjustment Layer**: Adjusts features to a fixed dimension.
- **Layer Normalization**: Stabilizes the training process.
- **Dropout Layer**: Prevents overfitting.
- **Fully Connected Layer**: Produces the binary classification output.

### Mixture of Experts (MoE) Model

- **Gating Network**: Processes embeddings from all experts, outputs gating weights.
- **Expert Networks**: Each processes its respective portion of the input embedding.
- **Weighted Combination**: Combines outputs from all experts to form a unified embedding.
- **Final Classifier**: Produces the binary classification output.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/Aakash-Tripathi/TFBS.git
    cd TFBS
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To train and evaluate the models, use the following commands:

1. **Train the models and Generate ROC curves**:
    ```sh
    python main.py
    ```

2. **Evaluate the models on OOD data**:
    ```sh
    python test.py
    ```

## Results

The results of the evaluation are presented using various metrics, including the Area Under the Curve (AUC). The MoE model generally outperforms individual expert models, demonstrating higher AUC scores and greater robustness.

### Statistical Analysis

- **Paired t-tests** and **ANOVA tests** were conducted to evaluate the statistical significance of performance differences between the expert models and the MoE model.
- Significant improvements in AUC scores were observed with the MoE model, confirming its enhanced generalization and robustness.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
