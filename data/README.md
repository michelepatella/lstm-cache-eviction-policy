---
annotations_creators:
- no-annotation
language:
- en
language_creators:
- other-none
license:
- mit
multilinguality:
- monolingual
pretty_name: Synthetic Data Accesses
size_categories:
- 100K<n<1M
source_datasets:
- original
tags:
- synthetic
- data-access-patterns
- workload-simulation
task_categories:
- tabular-classification
- time-series-forecasting
task_ids:
- tabular-multi-class-classification
- multivariate-time-series-forecasting
---

# Dataset Card for Synthetic Data Accesses

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage**: [N/A]
- **Repository:** https://github.com/michelepatella/lstm-cache-eviction-policy
- **Paper**: [N/A]
- **Leaderboard**: [N/A]
- **Point of Contact:** m.patella9@studenti.uniba.it

### Dataset Summary

This dataset collection provides synthetic data accesses. Each dataset contains 100,000 time-ordered events across a 100-key space, simulating complex data access behaviors. The collection includes both static (fixed key popularity) and dynamic (time-varying key popularity) variants.

### Supported Tasks and Leaderboards

`tabular-classification`, `time-series-forecasting`: This dataset collection can be used to train a model to predict which key will be accessed next in a synthetic workload sequence or to predict future sequences of accesses over time, capturing temporal patterns and trends. For classification tasks, each access event can be treated as a discrete class corresponding to the key. For forecasting tasks, sequences of past accesses can be used to predict future access patterns.

### Languages

Each dataset contains numeric values, while column names and documentation are in English (the associated BCP-47 code is `en`).

## Dataset Structure

### Data Instances

Each dataset row represents a single synthetic data access event, with a `timestamp` containing the hour of the day and a `request` corresponding to the accessed key.

An example from this dataset collection looks as follows:  
`{
  'timestamp': 11.977070545671724,
  'request': 58
}`


### Data Fields

Fields in raw datasets:
- `timestamp`:  Float, time of the data access event as the hour of the day [0.0,23.9]
- `request`: Integer, unique identifier of the requested key [1,100]

Additional fields in processed datasets:
- `sin_time`: Float, sine component of cyclical time [-1,1]
- `cos_time`: Float, cosine component of cyclical time [-1,1]
- `local_frequency`: Float, normalized short-term popularity of the current key [0,1]
- `local_recency`: Float, normalized short-term recency of the current key [0,1]

where the `timestamp` column is so replaced by its trigonometrical representation, while the target `request` column is preserved.

### Data Splits

No data splits are applied by default.

## Dataset Creation

### Curation Rationale

This dataset collection is created to provide a controlled environment for analyzing, modeling, and evaluating synthetic yet realistic data access patterns through simulation and experimentation, without relying on sensitive or proprietary real-world data.

### Source Data

#### Initial Data Collection and Normalization

Each dataset is synthetically generated to simulate realistic data access workloads. Two dataset variants are created:
  - _Static_: Fixed Zipf parameter, representing a moderate skew in key popularity
  - _Dynamic_: Zipf parameter varies linearly over time, introducing temporal variability in access distributions

Multiple time-dependent access patterns are applied to simulate realistic workloads, inspired by common usage behaviors:

- _Short-term repetition_: Repeated access to the same keys over short intervals to simulate local temporal locality
- _Oscillating toggles_: Alternating key requests between two base keys to simulate sudden demand shifts
- _Cyclic scanning_: Cyclical access of a subset of keys to reflect periodic workloads
- _Noisy distortion_: Random perturbations with occasional corrective accesses to mimic irregular or unpredictable behavior
- _Zipfian accesses_: Keys are sampled according to a Zipf distribution to capture popularity of hot keys
- _Memory effect_: Long-term repeated accesses with fallback to Zipf selection, simulating persistent popularity of some keys

Inter-request times are generated using an exponential distribution where the mean interval is determined by a hybrid model combining a cosine-based periodic function, and a burstiness component to simulate diurnal peaks in request rates.

Each raw dataset is preprocessed by removing missing values and engineering input features to capture temporal (`sin_time`, `cos_time`) and locality-aware (`local_frequency`, `local_recency`) access patterns.

#### Who are the source language producers?

No source language producers are involved.

### Annotations

#### Annotation process

No annotation processes are performed.

#### Who are the annotators?

[N/A]

### Personal and Sensitive Information

No personal and sensitive information is contained.

## Considerations for Using the Data

### Social Impact of Dataset

The purpose of this dataset collection is to help develop and evaluate models for predicting data access patterns in synthetic workloads.

A system that succeeds at the supported tasks would be able to accurately anticipate which key will be accessed next or forecast future sequences of accesses. The collection also serves as a controlled test-bed to benchmark models under various synthetic scenarios, enabling reproducible experiments and comparisons.

It should be noted, however, that all data is synthetically generated. This means the access sequences do not represent real user behavior or operational workloads, and any biases or limitations in the synthetic generation process may influence model performance.

### Discussion of Biases

- Data accesses follow a Zipf distribution, overrepresenting hot keys relative to less frequent ones
- Some data access patterns span longer time intervals, which may dominate a dataset relative to shorter patterns
- The collection favors regular, idealized data accesses over truly random or emergent behaviors

### Other Known Limitations

- The collection may not capture all nuances of real-world data access patterns, with rare behaviors potentially missing
- The limited key space (100 unique keys) may not reflect larger or more dynamic systems
- Inter-request times may not fully represent the complexity of real workloads
- Human behavior is only abstractly simulated and not directly represented

## Additional Information

### Dataset Curators

Michele Patella

### Licensing Information

MIT License

### Citation Information

@misc{synthetic_data_accesses_2026,
  author = {Michele Patella},
  title = {Synthetic Data Accesses},
  year = {2026},
  howpublished = {\url{https://github.com/michelepatella/lstm-cache-eviction-policy}}
}

### Contributions

Thanks to [@michelepatella](https://github.com/michelepatella) for adding this dataset collection.
