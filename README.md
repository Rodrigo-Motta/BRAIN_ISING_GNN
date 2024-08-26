# A Graph Neural Network Approach to Investigate Brain Critical States Over Neurodevelopment

## Overview

This repository contains the implementation and datasets for the paper "A Graph Neural Network Approach to Investigate Brain Critical States Over Neurodevelopment." The paper introduces a novel method to estimate the Ising Temperature of the brain from fMRI data using functional connectivity and Graph Neural Networks (GNNs). The main finding indicates a statistically significant negative correlation between age and Ising Temperature, providing insights into the brain's transition from a critical to a sub-critical state during neurodevelopment.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data Description](#data-description)
4. [Usage](#usage)
   - [Running Simulations](#running-simulations)
   - [Training the Graph Neural Network](#training-the-graph-neural-network)
   - [Estimating Brain Temperature](#estimating-brain-temperature)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgments](#acknowledgments)

## Introduction

This project uses Graph Neural Networks (GNNs) to analyze brain critical states during neurodevelopment. It focuses on estimating the Ising Temperature, a control parameter from the 2D Ising Model, using fMRI data. The approach leverages GNNs trained on simulated Ising Model networks to predict this parameter, shedding light on the dynamic changes in brain states as children mature.

## Installation

### Prerequisites

- Python 3.7 or later
- GeometricPyTorch
- Numpy
- Scipy
- Scikit-learn
- NetworkX

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/rodrigo-motta/brain-ising-gnn.git
    cd brain-ising-gnn
    ```

2. Install the required packages:
    ```bash
    pip install -r config/requirements.txt
    ```

## Data Description

The dataset used in this project is derived from the ADHD-200 Preprocessed repository, which includes resting-state fMRI data from Typically Developing Children and children with ADHD symptoms. The dataset has been preprocessed and parcellated into 333 cortical regions of interest (ROIs).

### Data Access

- Download the dataset from [here](http://fcon_1000.projects.nitrc.org/indi/adhd200/).
- The ADHD 4D is parcellated using 333 ROIs Gordon's Cortical Parcellation. This not available in the NITRC website, thus, if needed feel free to ask me.

## Usage

### Running Simulations

To generate Ising Model networks used for training:

```bash
python -m src.simulation.simulate_ising.py
 ```


### Training the Graph Neural Network

To generate Ising Model networks used for training:

```bash
python -m src.train.train_ising.py
```


### Estimating Brain Temperature

To generate Ising Model networks used for training:

```bash
python -m src.inference.fmri_temp.py
```

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes. For significant changes, please open an issue to discuss what you would like to change.

## License 

This project is licensed under the MIT License.

## Acknowledgments

This study was supported by the São Paulo Research Foundation (FAPESP) under various grants. Authors Rodrigo M. Cabral-Carvalho, Walter H. L. Pinaya, and João R. Sato.
