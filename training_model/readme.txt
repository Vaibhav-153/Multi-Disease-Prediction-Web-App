This folder contains the Jupyter Notebooks (.ipynb) used for training machine learning models in the project "Prediction of Disease Outbreaks Using Machine Learning."

Contents of This Folder:
  diabetes.ipynb – Notebook for training and evaluating the Diabetes Prediction Model.
  heart_disease.ipynb – Notebook for training and evaluating the Heart Disease Prediction Model.
  parkinsons.ipynb – Notebook for training and evaluating the Parkinson’s Disease Prediction Model.

Training Process (Inside Jupyter Notebooks):
  1. Data Preprocessing:
    Load the dataset and inspect its structure.
    Handle missing values, normalize data, and select relevant features.
  2. Model Training:
    Use Logistic Regression, Random Forest, and Support Vector Machines (SVM).
    Split the dataset into training and testing sets.
  3. Model Evaluation:
    Assess performance using accuracy, precision, recall, and F1-score.
    Compare different models to select the best one.
  4. Saving the Trained Model:
    The trained model is saved as a .sav file for later use in the web application.

How to Run These Notebooks?
  Open Jupyter Notebook or Google Colab.
  Load the required .ipynb file.
  Run the cells step by step to train and save the model.

Requirements:
Jupyter Notebook / Google Colab
Python 3.8+
Scikit-learn
Pandas
NumPy
Matplotlib
