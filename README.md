# Predicting_User_Coversion_On_MusicWebsite

## Project Overview
This is a project from UMN MSBA fall term, Business Analytics in R course.
This project builds a classification model to predict whether a free user will convert to a premium subscriber within the next six months after receiving targeted promotion.

In the project, the workflow of my analyses is below:
1. Define the business problem: identify likely adopters among free users.
2. Perform initial EDA and review the dataset structure.
3. Handle severe class imbalance using SMOTE after splitting the data into training, validation, and test sets.
4. Train and compare multiple models, including KNN, Decision Tree, Random Forest, and Ranger.
5. Evaluate model performance mainly with F1 score, with ROC/AUC as a secondary metric.
6. Select the best-performing model and apply it to final testing.

### Key Focus
This project demonstrates a full imbalanced-classification workflow, including preprocessing, oversampling, model comparison, threshold tuning, and performance evaluation.
