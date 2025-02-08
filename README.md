Prediction of Disease Outbreaks Using Machine Learning

Project Overview
  This project applies machine learning techniques to predict the likelihood of three major diseases:
    Diabetes
    Heart Disease
    Parkinson’s Disease
The goal is to develop an AI-driven web application that allows users to input their health parameters and receive real-time predictions, helping in early disease detection and risk assessment.

Project Structure

📂 Prediction-of-Disease-Outbreaks  
│── 📂 dataset             # Contains all CSV files (diabetes.csv, heart.csv, parkinsons.csv)  
│── 📂 trained_models      # Jupyter Notebooks (.ipynb) for training machine learning models  
│── 📂 saved_models        # Trained models saved as .sav files  
│── 📂 app                 # Contains web.py (main application file)  
│── 📄 README.md           # Project documentation and usage instructions  
│── 📄 requirements.txt    # List of required Python packages  

1️⃣ Datasets (dataset/)
  This folder contains medical datasets used for model training.

  diabetes.csv – Contains medical attributes for diabetes prediction.
  heart.csv – Includes key parameters like cholesterol, blood pressure for heart disease prediction.
  parkinsons.csv – Contains vocal and motor attributes for Parkinson’s detection.
  
2️⃣ Model Training (trained_models/)
  This folder contains Jupyter Notebook files for training machine learning models:
  
  diabetes.ipynb – Trains the Diabetes Prediction Model.
  heart_disease.ipynb – Trains the Heart Disease Prediction Model.
  parkinsons.ipynb – Trains the Parkinson’s Disease Prediction Model.
  Each notebook includes data preprocessing, model training, evaluation, and saving of trained models.

3️⃣ Trained Models (saved_models/)
  This folder stores pre-trained machine learning models in .sav format:
  
  diabetes_model.sav
  heart_disease_model.sav
  parkinsons_model.sav
  These models are used by the web application to provide real-time disease predictions.

4️⃣ Web Application (app/)
  This folder contains the Streamlit-based web app for real-time predictions.
  
  web.py – The main application script that loads trained models and takes user input.

To run the application, use:
streamlit run app/web.py

🔧 Installation & Setup
Step 1: Clone the Repository
  git clone https://github.com/your-username/Prediction-of-Disease-Outbreaks.git
  cd Prediction-of-Disease-Outbreaks
Step 2: Install Dependencies
  pip install -r requirements.txt
Step 3: Run the Web Application
  streamlit run app/web.py

📌 Features
  ✅ Machine Learning-Based Predictions (Diabetes, Heart Disease, Parkinson’s)
  ✅ User-Friendly Web App with Streamlit
  ✅ Trained Models Ready for Deployment
  ✅ Supports Real-Time Input & Predictions

📬 Contact
For any queries or contributions, feel free to reach out:
📧 Email: vnadamane15@gmail.com
🔗 Linkedin: www.linkedin.com/in/vaibhav-admane-15rtj
