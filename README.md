# CoSiTI 

## Community-Aware Item Similarity and Time Integration for Collaborative Filtering

## Overview

This repository contains the code for COSiTI: Community-Aware Item Similarity and Time Integration for Collaborative Filtering
. This project explores community aware enhancement of collaborative filtering models. It contains:
* Clustering code for community detection
* Baseline models where enhancement added 

## Getting Started

### Prerequisites

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data

In the folder data, run the `query_data_amazon_healthhousehold.py` or `query_data_amazon_fashion.py`, depending on which data you want to run the project on. The code returns a csv file with the relevant data for the models and clustering. Before running the data folder, download the Amazon reviews data from https://amazon-reviews-2023.github.io/ and add into the folder. You will need both the metadata and the review data. 

### Running the clustering

In order to run the clustering approach navigate to `clustering/community_detection_amazon.py` and add the relevant data to cluster. The code outputs plots of the top clusters, the cluster evaluation score and the labels from the clusters/communities to relevant training data for the model, 

### Running the models

To run the models, locate the relevant folder and run the desired model:

```bash
# E.g. Ablation item and time
python run model_ncf_ablation_amazon.py

# E.g. combined (CoSiTI approach)
python run model_ncf_combined_analysis.py
```

All models will output the average over X runs (currently set to 3, can change yourself) and save the results down to a CSV file for further analysis. The analysis and combined files produce a CSV file with a grid results showing how the values differ with alpha.

## Project Structure

```
cositi/
│
├── data/                            # Data acquisition scripts
│   ├── query_data_amazon_healthhousehold.py          # Script to preprocess Amazon Health & Household dataset
│   └── query_data_amazon_fashion.py           # Script to preprocess Amazon Health & Household dataset
│
├── clustering/                      # Community detection algorithms
│   └── community_detection_amazon.py # Main clustering code
│
├── models/                          # All baseline recommendation models               
│   ├── bpr/                        # Bayesian Personalized Ranking models
│   │   ├── model_bpr.py            # Basic BPR model implementation
│   │   ├── model_bpr_item_analysis.py  # Item-similarity enhanced
│   │   ├── model_bpr_time_analysis.py  # Time-integration enhanced
│   │   └── model_bpr_combined_ikea.py  # Full CoSiTI implementation
│   │
│   ├── ncf/                        # Neural Collaborative Filtering models
│   │
│   ├── mf/                         # Matrix Factorization models
│   │
│   └── neumf/                      # Neural Matrix Factorization models
│   
│
├── main.py                          # Main execution script (WIP)
├── Makefile                         # Automation commands (WIP)
├── requirements.txt                 # Project dependencies
└── README.md                        # This file
```

### Clustering

In the clustering folder we find the code to run the clustering approach and visualize it + connect back to the data for model training.

### Model Code

All baseline models, to which I add CoSiTi can be found under:

```
models
```

## Key Findings

Our experiments demonstrate that CoSiTI consistently outperforms baseline collaborative filtering models and shows competitive performance against specialized context-aware models.
