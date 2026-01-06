---
annotations_creators:
- no-annotation
language:
- en
language_creators:
- machine-generated
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
- time-series
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

- **Homepage**: -
- **Repository:** https://github.com/michelepatella/lstm-cache-eviction-policy
- **Paper**: -
- **Leaderboard**: -
- **Point of Contact:** m.patella9@studenti.uniba.it

### Dataset Summary

The dataset collection provides synthetic data accesses designed for benchmarking, modeling, and analysis of realistic workload behaviors. Each dataset contains 100,000 time-ordered events across a 100-key space, simulating complex data access behaviors. The dataset collection includes both static (fixed key popularity) and dynamic (time-varying key popularity) variants.

### Supported Tasks and Leaderboards

- Analysis and study of temporal patterns in sequences of data accesses
- Modeling and simulation of data access behaviors
- Evaluation of algorithms operating on time-ordered sequences
- Study of frequency and popularity of data in request series
- Generation of synthetic scenarios for controlled experiments
- Experimentation with time series prediction and analysis methods
- Support for research on short- and long-term memory effects in access patterns
- Development and validation of optimization techniques on temporally ordered data

### Languages

Each dataset contains numeric values, while column names and documentation are in English.

## Dataset Structure

### Data Instances

Each dataset is a CSV file which consists of 100,000 time-ordered rows, each representing a single synthetic data access event.

Example:
| timestamp          | request |
|--------------------|---------|
|       ...          | ...     |
| 11.977070545671724 | 58      |
| 11.980916337369722 | 54      |
| 11.993804355974053 | 59      |
| 11.995151057904133 | 55      |
| 12.002285716536916 | 56      |
| 12.006074313768924 | 61      |
| 12.012448374092424 | 57      |
| 12.012770222412232 | 62      |
|       ...          | ...     |

### Data Fields

Fields in raw datasets:
- `timestamp`:  Float, time of the data access event as the hour of the day [0.0,23.9]
- `request`: Integer, unique identifier of the requested key [1,100]

Additional fields in processed datasets:
- `sin_time`: Float, sine component of cyclical time [-1,1]
- `cos_time`: Float, cosine component of cyclical time [-1,1]
- `local_frequency`: Float, normalized short-term popularity of each key [0,1]
- `local_recency`: Float, normalized recency of each key [0,1]

where the `timestamp` column is so replaced by its trigonometrical representation, while the target `request` column is preserved.

### Data Splits

No data splits are applied.

## Dataset Creation

### Curation Rationale

The dataset collection is created to provide a controlled environment for analyzing, modeling, and evaluating synthetic yet realistic data access patterns through simulation and experimentation, without relying on sensitive or proprietary real-world data.

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

No annotators are involved.

### Personal and Sensitive Information

No personal and sensitive information is contained.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset collection is intended for research, benchmarking, and simulation. Since it is fully synthetic, it does not reflect real user behavior or contain personal information, and its use poses minimal social impacts.

### Discussion of Biases

- Data accesses follow a Zipf distribution, overrepresenting “hot” keys relative to less frequent ones
- Some data access patterns span longer time intervals, which may dominate a dataset relative to shorter patterns
- The collection favors regular, idealized data accesses over truly random or emergent behaviors

### Other Known Limitations

- The dataset collection may not capture all nuances of real-world data access patterns, with rare behaviors potentially missing
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
