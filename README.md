# Managing-AI-Development
# Week 2 AI Data Preprocessing Exercise

## Overview
This is a repository for an exercise related to Lecture 2: Data Analysis and Feature Engineering for AI Applications. This exercise aims at applying the entire early machine learning pipeline with the use of structured data.
## Dataset
Dataset: UCI Wine Quality Dataset. It contains features of wine samples with respect to their physicochemical properties, as well as a quality score. In this exercise, the problem is to predict the quality of a wine sample given its features.
## Project Structure
- `data/`: holds the dataset or its link to download it.
- `src/train_model.py`: loads the dataset, pre-processes features, trains several models, and prints evaluation metrics.
- `outputs/`: holds generated plots and evaluation metrics.
- `requirements.txt`: holds list of required Python packages to run this project.
## How to run
1. Install Python version 3.10 or higher.
2. Create a virtual environment.
3. Run `pip install -r requirements.txt`.
4. Put dataset CSV file in `data/` directory.
5. Run `python src/train_model.py`.

## Expected Output
The script prints missing value checks, dataset shape, model performance scores, and a confusion matrix. It also saves simple exploratory plots in the `outputs/` folder.
## Learning Outcomes
By completing this exercise, students practise data loading, exploration, feature scaling, train-test splitting, model comparison, and evaluation using metrics beyond simple accuracy.
