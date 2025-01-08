# Tree Methods Consulting Project with PySpark

This project implements a Random Forest model to identify which chemical preservative is causing early spoilage in dog food batches. This exercise is part of the course "Spark and Python for Big Data with PySpark".

## Project Overview

You have been hired by a dog food company to predict why some batches of their dog food are spoiling much faster than expected. The company has not updated its machinery, which means the amounts of the five chemical preservatives they are using can vary greatly. But which chemical has the strongest effect?

The dog food company mixes a batch of preservatives containing 4 different chemical preservatives (A, B, C, D) and then completes it with a "filler" chemical. Food scientists believe that one of the preservatives A, B, C, or D is causing the problem, but they need you to figure out which one!

Use Machine Learning with Random Forest to discover which parameter has the greatest predictive power, thus identifying which chemical causes early spoilage. Then, find out how you can decide which chemical is the problem.

## Data Description

The dataset contains information about the percentages of four chemical preservatives in the dog food mixture and whether the batch spoiled or not. The data is saved in a CSV file called `dog_food.csv`.

### Variables/Columns

- **A**: Integer - Percentage of preservative A in the mixture
- **B**: Integer - Percentage of preservative B in the mixture
- **C**: Double - Percentage of preservative C in the mixture
- **D**: Integer - Percentage of preservative D in the mixture
- **Spoiled**: Double - Label indicating whether the batch of dog food spoiled or not

## Objective

The objective of this project is to create a classification model that helps identify which chemical preservative is causing early spoilage in dog food batches.

## Methodology

The project uses the following steps:

1. **Data Loading and Preparation**:
    - The data is loaded from a CSV file into a Spark DataFrame.
    - The schema of the data is printed to ensure correct data loading.
2. **Feature Engineering**:
    - The numerical features are assembled into a single feature vector using `VectorAssembler`.
3. **Random Forest Model**:
    - A Random Forest model is trained using the entire dataset.
    - The model is trained using the `Spoiled` column as the label.
4. **Model Evaluation**:
    - The model's feature importances are evaluated to determine which chemical preservative has the greatest predictive power.

## Results

The model was evaluated using the feature importances, and the following results were obtained:

- **Feature Importances**: The importance of each chemical preservative in predicting spoilage.
- **Predominant Chemical**: The chemical preservative C has the greatest predictive power, indicating it is the most likely cause of early spoilage.

## Conclusion

The Random Forest model helps identify which chemical preservative is causing early spoilage in dog food batches. The feature importances provide insights into which chemical has the strongest effect.

## Repository Structure

- `Tree_Methods_Consulting_Project.ipynb`: Jupyter notebook containing the entire project code and analysis.
- `dog_food.csv`: Dataset used for the project.
- `tree_methods_consulting_project.py`: Python script with cleaned project code.

## Usage

To run this project, ensure that you have PySpark installed and that you have access to the dataset `dog_food.csv`. You can run the code using the Jupyter notebook `Tree_Methods_Consulting_Project.ipynb`.

1. **Set Up:** Ensure you have a Spark environment running.
2. **Data:** Place the `dog_food.csv` in the same directory as the notebook or adjust the file path in the code.
3. **Run:** Execute the Jupyter notebook.

## Code and Libraries

The code is implemented using PySpark and the following libraries:

- `pyspark.sql.SparkSession`: For Spark session management.
- `pyspark.ml.feature.VectorAssembler`: For assembling features into a single vector.
- `pyspark.ml.classification.RandomForestClassifier`: For the Random Forest model.
