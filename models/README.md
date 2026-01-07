---
language:
- en
license: mit
library_name: pytorch
tags:
- lstm
- next-key-access-prediction
- time-series
datasets:
- synthetic_data_accesses
metrics:
- accuracy
- precision_macro
- precision_weighted
- recall_macro
- recall_weighted
- f1_macro
- f1_weighted
- cohen_kappa
base_model: lstm-base
model-index:
- name: static_lstm
  results:
  - task:
      type: next-key-access-prediction
      name: Next-Key Access Prediction
    dataset:
      type: synthetic_data_accesses
      name: Synthetic (Static) Data Accesses
      config: static
      split: test
      revision: 1.0
      args: {}
    metrics:
      - type: accuracy
        value: 0.88
        name: Static Model Accuracy
      - type: precision_macro
        value: 0.93
        name: Static Model Macro Precision
      - type: precision_weighted
        value: 0.89
        name: Static Model Weighted Precision
      - type: recall_macro
        value: 0.92
        name: Static Model Macro Recall
      - type: recall_weighted
        value: 0.88
        name: Static Model Weighted Recall
      - type: f1_macro
        value: 0.92
        name: Static Model Macro F1
      - type: f1_weighted
        value: 0.88
        name: Static Model Weighted F1
      - type: cohen_kappa
        value: 0.88
        name: Static Model Cohen’s Kappa
    source:
      name: Michele Patella
      url: https://github.com/michelepatella/lstm-cache-eviction-policy
- name: dynamic_lstm
  results:
  - task:
      type: next-key-access-prediction
      name: Next-Key Access Prediction
    dataset:
      type: synthetic_data_accesses
      name: Synthetic (Dynamic) Data Accesses
      config: dynamic
      split: test
      revision: 1.0
      args: {}
    metrics:
      - type: accuracy
        value: 0.85
        name: Dynamic Model Accuracy
      - type: precision_macro
        value: 0.88
        name: Dynamic Model Macro Precision
      - type: precision_weighted
        value: 0.86
        name: Dynamic Model Weighted Precision
      - type: recall_macro
        value: 0.88
        name: Dynamic Model Macro Recall
      - type: recall_weighted
        value: 0.85
        name: Dynamic Model Weighted Recall
      - type: f1_macro
        value: 0.88
        name: Dynamic Model Macro F1
      - type: f1_weighted
        value: 0.85
        name: Dynamic Model Weighted F1
      - type: cohen_kappa
        value: 0.84
        name: Dynamic Model Cohen’s Kappa
    source:
      name: Michele Patella
      url: https://github.com/michelepatella/lstm-cache-eviction-policy
---

# Model Card for LSTM-Based Data Access Predictor

LSTM-based models for predicting next-key accesses in synthetic workload sequences.

## Model Details

### Model Description

This collection of LSTM models captures temporal dependencies in synthetic key access sequences to predict future accesses. Each model is trained on either static or dynamic datasets, leveraging temporal-aware loss and class-balanced weighting. The models are optimized with pruning and quantization for efficient inference and are intended for research, benchmarking, and educational purposes in sequence modeling and time-series prediction.

- **Developed by:** Michele Patella
- **Model type:** Recurrent Neural Network (RNN)
- **Language(s) (NLP):** -
- **License:** MIT
- **Finetuned from model:** -

### Model Sources

- **Repository:** https://github.com/michelepatella/lstm-cache-eviction-policy
- **Paper:** -
- **Demo:** -

## Uses

### Direct Use

This LSTM collection is intended for research and experimentation on synthetic workload sequences. It can be used to:

- Make predictions for all possible keys in a sequence of synthetic data accesses
- Generate future sequences of predictions for key accesses using an autoregressive rollout
- Benchmark sequence modeling approaches on controlled synthetic workloads
- Support educational or proof-of-concept experiments in sequential modeling and time-series prediction

### Downstream Use

This LSTM collection can serve as a base for downstream tasks in sequence modeling and time-series prediction, specifically in synthetic or simulated environments. Potential uses include:

- Fine-tuning on specific synthetic workloads with different settings
- Using predictions in simulators for synthetic workload scenarios
- Integrating into educational tools to demonstrate sequence modeling and probabilistic predictions
- Experimenting with extensions

### Out-of-Scope Use

This LSTM collection is not intended for production workloads or real-world key access prediction. Specifically, it should not be used for:

- Predicting access patterns in real databases, caches, or systems
- Making high-stakes operational decisions in production environments
- Security-critical applications or any domain requiring high reliability
- Any use outside synthetic or controlled experimental settings

## Bias, Risks, and Limitations

_Bias_:
- Some keys are more frequent, and longer patterns may dominate, biasing predictions
- Sociotechnical biases don’t apply to synthetic data, but real-world use may amplify unrepresentative patterns

_Risks_:
- Using this model collection for production workloads can lead to poor predictions
- Autoregressive sequence generation may accumulate errors over long sequences
- The collection is not safety-critical; it is unsuitable for decisions requiring high reliability
- The collection is not trained on real data; it may behave unpredictably on sensitive or operational inputs

_Limitations_:
- The collection does not generalize to real-world key access patterns
- Limited vocabulary of 100 keys restricts the diversity of predictions
- Autoregressive rollout must be implemented separately for multistep predictions
- Evaluation on downstream tasks is limited to synthetic or simulated environments

### Recommendations

- Do not use this model collection for production workloads or operational decision-making
- When generating sequences autoregressively, validate results carefully
- Avoid applying a model to real-world or sensitive data without proper evaluation
- Use this model collection primarily for research, educational purposes, and controlled experiments with synthetic data

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import torch

# Set QNNPACK as quantization engine
torch.backends.quantized.engine = "qnnpack"

# Define a device to run computations on
device = "cpu"

# Load pretrained model (full class)
model = torch.load(
    "path/to/pretrained_model.pt",
    weights_only=False,
)

# Prepare model for inference: set it to
# evaluation mode and move it to specified device
model.eval()
model.to(device)

# Prepare an example input of length 25
x_features = torch.tensor(
    [
        [0.6223982138696689, -0.7827007495664265, 0.04, 0.0],
        [0.6124288405323478, -0.7905257208239362, 0.04, 0.0],
        [0.6104719796770406, -0.7920378539117906, 0.04, 0.0],
        [0.6097757808312311, -0.7925739694890708, 0.04, 0.0],
        [0.6066941192900404, -0.7949353719761649, 0.04, 0.0],
        [0.6006270322950682, -0.7995293415981799, 0.04, 0.0],
        [0.6002082961660248, -0.7998437354968015, 0.04, 0.0],
        [0.5998344491575055, -0.8001241363712958, 0.04, 0.0],
        [0.5938129235087608, -0.8046031393637354, 0.04, 0.0],
        [0.5925081773685348, -0.8055644355055763, 0.04, 0.0],
        [0.5911324085641365, -0.8065745319219815, 0.04, 0.0],
        [0.5902152470971341, -0.8072459117852929, 0.04, 0.0],
        [0.5897141331658644, -0.8076120610444306, 0.04, 0.0],
        [0.5868317927790053, -0.8097088655706936, 0.04, 0.0],
        [0.5788425550794389, -0.8154393272519462, 0.04, 0.0],
        [0.5700334592950794, -0.8216214793224953, 0.04, 0.0],
        [0.5631242943025839, -0.8263722098220734, 0.04, 0.0],
        [0.5578902298076331, -0.8299147495286406, 0.04, 0.0],
        [0.5509957413432837, -0.8345080544977174, 0.04, 0.0],
        [0.5502643809076128, -0.8349904856370292, 0.04, 0.0],
        [0.5491437378353997, -0.8357279193590255, 0.04, 0.0],
        [0.5490740116917225, -0.8357737311526117, 0.04, 0.0],
        [0.5412869715427089, -0.8408379240008877, 0.04, 0.0],
        [0.5380683934051343, -0.8429011828313077, 0.04, 0.0],
        [0.5360692407649906, -0.8441740158910641, 0.04, 0.0],
    ],
    dtype=torch.float,
    device=device,
)
x_keys = torch.tensor(
    [
        [
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
        ]
    ],
    dtype=torch.long,
    device=device
)

# Compute logits for all keys for the next step
# 1) These logits can be used for downstream tasks
# 2) Multistep predictions can be obtained via autoregressive rollout by feeding
# the model’s outputs back as inputs iteratively
predictions = model(x_features, x_keys)

print("Model predictions:", predictions)
```

## Training Details

### Training Data

This LSTM collection is trained on a collection of synthetic workload datasets, each containing 100,000 sequences of key accesses over a 100-key space.

Each model is trained exclusively on its corresponding processed variant:
- _Static model_: Trained on 80% of the static dataset
- _Dynamic model_: Trained on 80% of the dynamic dataset

For full details, refer to the [Dataset Card](../data/README.md).

### Training Procedure

#### Preprocessing

The synthetic workload sequences are preprocessed before being fed to each LSTM model. Missing values are removed, and timestamps are encoded trigonometrically as sine and cosine values in [-1,1] to preserve cyclical time. Two locality-based features are computed over a fixed-length rolling window: local frequency, which captures short-term key popularity, and local recency, which reflects recent accesses; both are normalized to [0,1]. Target keys are embedded into 32-dimensional vectors and concatenated with the feature tensor, producing a model-ready input. Finally, sequences are truncated or padded to a fixed length of 25.

#### Training Hyperparameters

- **Training regime:** fp32

| **Hyperparameter**    | **Static** | **Dynamic ** |
|-----------------------|------------|--------------|
| _**Dataset**_         |
| Size (validation set) | 0.2        | 0.2          |
| Size (training set)   | 0.8        | 0.8          |
| _**Data Loader**_     |
| Batch size            | 512        | 512          |
| Shuffle               | False      | False        |
| _**Early Stopping**_  |
| Delta (training)      | 0.0001     | 0.0001       |
| Delta (validation)    | 0.0005     | 0.0005       |
| Patience (training)   | 10         | 10           |
| Patience (validation) | 3          | 3            |
| _**Loss**_            |
| Class weights         | Balanced   | Balanced     |
| Reduction             | None       | None         |
| _**Model**_           |
| Batch first           | True       | True         |
| Bias                  | False      | False        |
| Bidirectional         | False      | False        |
| Dropout               | 0.1        | 0.1          |
| Embedding dimension   | 32         | 32           |
| Hidden layers         | 2          | 2            |
| Hidden size           | 256        | 256          |
| Projection size       | 0          | 0            |
| Sequence length       | 25         | 25           |
| _**Optimizer**_       |
| Learning rate         | 0.001      | 0.005        |
| Type                  | AdamW      | AdamW        |
| Weight decay          | 0.05       | 0.05         |
| _**Pruning**_         |
| Amount                | 0.2        | 0.2          |
| _**Quantization**_    |
| Engine                | QNNPACK    | QNNPACK      |
| Type                  | qint8      | qint8        |
| _**Resources**_       |
| Cores                 | 4          | 4            |
| Device                | CPU        | CPU          |
| _**Seed**_            |
| Value                 | 42         | 42           |
| _**Training**_        |
| Epochs                | 500        | 500          |
| _**Validation**_      |
| Epochs                | 5          | 5            |
| Folds                 | 5          | 5            |

#### Speeds, Sizes, Times

| **Metric**             | **Static**        | **Dynamic**       |
|------------------------|-------------------|-------------------|
| _**Speeds**_           |
| Batches per epoch      | ~195              | ~195              |
| Inference throughput   | ~25,000 seq/h     | ~25,000 seq/h     |
| Sequences per epoch    | 100,000           | 100,000           |
| Training throughput    | ~14,000,000 seq/h | ~13,600,000 seq/h |
| _**Sizes**_            |
| Dataset size           | 5.44 MB           | 5.43 MB           |
| Model parameters       | ~269,824          | ~269,824          |
| Model size             | 3.3 MB            | 3.3 MB            |
| _**Times**_            |
| Epochs                 | 35                | 45                |
| Training time          | ~15min            | ~20min            |
| Validation time        | ~3h               | ~3h               |

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

This LSTM collection is tested on a collection of synthetic workload datasets, each containing 100,000 sequences of key accesses over a 100-key space.

Each model is tested exclusively on its corresponding processed variant:
- _Static model_: Tested on 20% of the static dataset
- _Dynamic model_: Tested on 20% of the dynamic dataset

For full details, refer to the [Dataset Card](../data/README.md).

#### Factors

The evaluation of this LSTM collection is disaggregated by dataset variant (static vs dynamic).

#### Metrics

The evaluation of this LSTM collection uses the following metrics:

- _Accuracy_: Fraction of correct predictions over all samples
- _Precision (macro & weighted)_: Measures the correctness of positive predictions, averaged across classes
- _Recall (macro & weighted)_: Measures the coverage of true positives, averaged across classes
- _F1 score (macro & weighted)_: Harmonic mean of precision and recall, balancing both metrics
- _Cohen’s Kappa_: Measures agreement between predicted and true labels, accounting for chance agreement

### Results

| **Dataset** | **Macro (Precision, Recall, F1)** | **Weighted (Precision, Recall, F1)** | **Accuracy** | **Cohen’s Kappa** |
|-------------|-----------------------------------|--------------------------------------|--------------|-------------------|
| Static      | 0.93, 0.92, 0.92                  | 0.89, 0.88, 0.88                     | 0.88         | 0.88              |
| Dynamic     | 0.88, 0.88, 0.88                  | 0.86, 0.85, 0.85                     | 0.85         | 0.84              |

#### Summary

The LSTM models achieve high performance on next-key prediction in synthetic workloads. The static model slightly outperforms the dynamic model across all metrics, reflecting the greater stability of static access patterns. Overall, macro and weighted precision, recall, and F1 scores are consistently above 0.85, with Cohen’s Kappa and accuracy closely matching, indicating reliable predictive performance in controlled synthetic workloads.

## Model Examination

Analysis of models behavior reveals the following patterns:
- Fewer errors occur on frequent keys; rare keys (tail of the Zipf distribution) have higher error rates
- Simple patterns (e.g., short-term repetition) are predicted more accurately; complex patterns (e.g., toggles) cause more mistakes
- The sequence of recently accessed keys strongly influences predictions
- In autoregressive rollout, errors tend to accumulate for steps further in the future; recent steps are predicted better
- Dynamic Zipf distributions increase difficulty, producing more errors

## Environmental Impact

- **Hardware Type:** Apple M2 Chip
- **Hours used:** 6.6h
- **Cloud Provider:** -
- **Compute Region:** Italy
- **Carbon Emitted:** 163.92 gCO₂e

## Technical Specifications

### Model Architecture and Objective

This model collection implement LSTM networks to predict future key accesses, capturing mid- and long-term temporal dependencies while mitigating exploding/vanishing gradients.

Architecture highlights:
- Two unidirectional hidden layers, 256 units each, with dropout 0.1
- Input sequences of length 25, combining target key embeddings (32-dimensional) with engineered temporal and locality features
  - Resulting input tensor shape: [512, 25, 36] (batch size, sequence length, embedded keys + features concatenated)
- Sigmoid activations for input, forget, and output gates; tanh for cell and hidden states
- Output logits for 100 keys via fully connected layer from the last hidden state
  - Resulting output tensor shape: [512, 100] (batch size, number of classes)

Training objective:
- Temporal-aware Cross-Entropy loss, prioritizing accurate predictions for imminently accessed data items
- Class-balanced weighting to handle Zipf-distributed access skew

### Compute Infrastructure

The LSTM models were trained and evaluated using a dedicated local Apple Silicon environment. The pipeline leverages Python Multiprocessing, CPU parallelization (via Ray), and PyTorch Distributed Data Processing (DDP) to speed up computations.

#### Hardware

- **CPU:** Apple M2 Chip (4 cores utilized)
- **Memory:** 8 GB
- **Architecture:** ARM64

#### Software

- **OS:** macOS 26.1
- **Programming Language:** Python (3.12.11)
- **Model Framework:** PyTorch (2.8.0)
- **Parallelization Frameworks**: Ray (2.52.1), PyTorch DDP, and Multiprocessing library

## Citation

No papers or posts introducing the model collection to cite.

**BibTeX:**

[N/A]

**APA:**

[N/A]

## Glossary

No additional terms or calculations.

## More Information

No additional information.

## Model Card Authors

Michele Patella

## Model Card Contact

m.patella9@studenti.uniba.it
