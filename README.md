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