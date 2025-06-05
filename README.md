# DataWhisperers

## Introduction
This repository contains the code produced during the EPFL course Machine Learning for Behavioral Data, 
taken during the Spring 2025 semester. 

In this project, we explored unsupervised learning techniques
to uncover behavioral personas from the Lernnavi platform,
a digital tool designed to support foundational skills development among secondary school students in Switzerland.
Our goal was to move from raw interaction data to interpretable
learner archetypes that reflect both static engagement styles
and evolving behavioral trajectories.

This repository contains all the code required to analyze the data and visualize the discovered learner personas and their behavioral patterns.

## Requirements
To be able to run the code, two important steps are needed.

First, the data should be in the ```/data/Lernnavi``` folder. It should contains tables named ```users.csv.gz```, ```events.csv.gz```, and ```transactions.csv.gz```.

Then, the required packages are listed in ```requirements.txt```. They can be installed using:
```
pip install -r requirements.txt
```

## Structure

You can find all results in the file ```m7_DataWhisperers.ipynb```.

You can look at the results directly in the notebook. You can run it but it takes a large amount of time

If you want to run the notebook, we advise you to run ```m4_DataWhisperers.ipynb``` and ```m6_DataWhisperers.ipynb``` separately for the sake of your computer.
```
.
├── data
│   └── Lernnavi
│       ├── 2025-MLBD Lernnavi Data Description.docx
│       ├── documents.csv.gz
│       ├── events.csv.gz 
│       ├── feedback.csv.gz
│       ├── study
│       │   ├── 2025-MLBD Lernnavi Study Data Description.docx
│       │   ├── events.csv.gz
│       │   ├── math_prepost_test.csv
│       │   ├── postsurvey_formatted.csv
│       │   ├── postsurvey_questions.txt
│       │   ├── presurvey_formatted.csv
│       │   ├── presurvey_questions.txt
│       │   └── transactions.csv.gz
│       ├── topics_translated.csv
│       ├── topic_trees.csv.gz
│       ├── transactions.csv.gz
│       └── users.csv.gz
├── df_q1.npy
├── df_q2.npy
├── features_final.csv
├── final_features_for_clustering.csv
├── final_scaled_features_with_userid.csv
├── m4_DataWhisperers.ipynb                 <--- Notebook containing our work on HDBSCAN and GMM clustering
├── m6_DataWhisperers.ipynb                 <--- Notebook containing our work on DTW clustering and our LSTM try
├── m7_DataWhisperers.ipynb                 <--- Main notebook containing all our work
├── README.md
├── requirements.txt
├── scaler.pkl
└── utils
    ├── clustering_eval_utils.py
    ├── clustering.py
    ├── feature_processing.py
    └── memory_usage.py
```