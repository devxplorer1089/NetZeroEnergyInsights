# Net-Zero Buildings Data Challenge – EU Edition

## 📌 Overview

This repository contains the code, data, and analysis for the Net-Zero Buildings Data Challenge – EU Edition. The goal of this challenge was to analyze energy performance data, identify cost-effective renovation strategies, and develop machine learning models to optimize energy efficiency in buildings.

I focused on the France dataset, performing Exploratory Data Analysis (EDA), building predictive models, and deriving actionable insights to improve energy efficiency.

## 🔹 Key Files

- data/ → Contains raw and cleaned datasets.

- notebooks/dpe_france.ipynb → Main notebook for data exploration and model development.

- scripts/01_clean_dataset.py → Cleans and preprocesses the dataset.

- scripts/02_normalize_data.py → Normalizes numerical features for better model performance.

- scripts/03_train_xgb_model.py → Trains an XGBoost model for energy efficiency classification.

- plots/ → Contains PNG files of key visual insights.

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to uncover key insights regarding:

- Energy Efficiency Distribution 📉

- Building Elements Impact on Energy Use 🏠

- Cost-Effective Renovation Strategies 💰

- Financial Efficiency of Measures 💲

- Correlation of Features 🔗

- Energy Consumption Trends by Department 📍

## 🤖 Machine Learning Model

We developed an XGBoost-based model for energy efficiency classification. The model helps in:

- Predicting energy efficiency ratings based on building attributes.

- Identifying optimal renovation measures to improve energy performance.

### 🔹 Model Performance

- Accuracy: XX%

- Feature Importance: Key features influencing energy efficiency include insulation, heating system, and building age.

## 🏆 Evaluation Criteria

The project was structured to maximize points based on:

- Exploratory Data Analysis (30 pts) ✔️

- Algorithm Performance (40 pts) ✔️

- Report Quality (20 pts) ✔️

- Bonus: Real-World Application (10 pts) ✔️

## 🛠 How to Run the Code

### 1️⃣ Clone the Repository

git clone https://github.com/yourusername/netzero_buildings_france.git
cd netzero_buildings_france

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Run the Scripts

- Preprocess the data:

python scripts/01_clean_dataset.py

- Normalize the data:

python scripts/02_normalize_data.py

- Train the model:

python scripts/train_xgb_model.py

- Generate visualizations: Run the Python File scripts/04_generate_visuals.py
