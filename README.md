## 🌐 **Live Demo:** https://cardiovision-ai-sahil.streamlit.app
# 🫀 CardioVision AI
### AI-Powered Heart Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)
![ML](https://img.shields.io/badge/Machine%20Learning-6%20Models-green)
![Accuracy](https://img.shields.io/badge/F1%20Score-99.14%25-brightgreen)

---
## 📌 Overview

CardioVision AI is an AI-powered heart disease prediction system built using Python and Streamlit. It compares six machine learning models and automatically selects the best-performing model (XGBoost) for real-time prediction.
The system supports single-patient prediction, bulk CSV prediction, what-if simulation, AI assistance, patient history tracking, and medical report scanning.

## ✨ Features
- 🔬 Single Patient Prediction using 6 ML models
- 📊 What-If Simulation with live risk gauge (0–100)
- 📁 Bulk Prediction via CSV upload
- 🕐 Patient History with risk trend tracking
- 🤖 AI Assistant powered by Google Gemini
- 📄 Medical Report Scanner (OCR)
- 📈 Model Metrics Dashboard (Simple & Technical Views)
- 🏆 Automatic Best Model Selection

## 🔄 Workflow
1. User enters patient data or uploads a medical report
2. Data is preprocessed and validated
3. Multiple ML models generate predictions
4. XGBoost is selected as the primary prediction model
5. Results and risk insights are displayed
6. Patient history is stored for future analysis

## 🎯 Model Performance
Model	                         F1 Score	              ROC-AUC
XGBoost (Best)	                0.9914                	0.9991
Logistic Regression           	0.9871	                0.9982
Random Forest	                  0.9871                 	0.9992
Neural Network	                0.9829                	0.9936
SVM                            	0.9702	                0.9969
Decision Tree	                  0.9697	                0.9834

Best Model: XGBoost (F1 Score: 99.14% | ROC-AUC: 99.91%)

## 🛠️ Tech Stack

- Language: Python
- Framework: Streamlit
- Machine Learning: XGBoost, Random Forest, SVM, Logistic Regression, Decision Tree, Neural Network
- Libraries: Scikit-learn, Pandas, NumPy
- AI Integration: Google Gemini API
- Database: SQLite

## 📊 Dataset
- UCI Heart Disease Dataset
- 303 Patients
- 14 Clinical Features

## 🚀 How to Run
```bash
git clone https://github.com/sahilbetalcs-byte/CardioVision-AI
cd CardioVision-AI
pip install -r requirements.txt
streamlit run app.py
```

## 📷 Screenshots
 1. Heart Disease Prediction Input Screen
<img width="1176" height="876" alt="Fig_01_Heart_Disease_Prediction_Input_Screen" src="https://github.com/user-attachments/assets/e6cbfbf8-f1dc-42ee-a053-e7c875682ede" />

 2. What-If Risk Simulation Dashboard
<img width="1281" height="777" alt="Fig_02_Heart_Disease_Prediction_What_If_Simulation" src="https://github.com/user-attachments/assets/731da7b0-342f-4cbd-b4d5-3fbbf92b0f06" />


 3. Bulk Prediction Module
<img width="1301" height="521" alt="Fig_3_Bulk_Prediction_Module" src="https://github.com/user-attachments/assets/5415ed04-8a82-4ca1-b369-a2d445e13e17" />

4. Patient History & Risk Tracking
<img width="1305" height="838" alt="Fig_04_Patient_History_Module" src="https://github.com/user-attachments/assets/770ef71b-19e8-4038-99c1-96e1d4c81bf3" />

 5. AI Medical Assistant
<img width="1285" height="668" alt="Fig_05_AI_Assistant_Modulo" src="https://github.com/user-attachments/assets/67b5bee5-7383-4ccc-b0d1-83c68545d0d0" />

 6. Model Metrics Dashboard
<img width="1270" height="806" alt="Fig_06_Model_Metrics_Dashboard" src="https://github.com/user-attachments/assets/5a46afd5-9293-45f0-a9cc-65c3462f82a9" />

7. Detailed Model Performance Analysis
<img width="1248" height="901" alt="Fig_07_Detailed_Model_Performance_Analysis" src="https://github.com/user-attachments/assets/181e95a2-0c3b-40ff-9430-f118890126a5" />

8. Comparative Model Evaluation
<img width="1297" height="697" alt="Fig_08_Detailed_Model_Evaluating" src="https://github.com/user-attachments/assets/09e3b5d8-2509-4a45-b087-a6473e277f24" />

## 👨‍💻 Developer

** Sahil Betal **

B.Tech Computer Technology (2026)
KITS Ramtek, Nagpur
GATE CSE 2026 Qualified
Aspiring Data Analyst & ML Engineer
Project Leader (Team of 5)

## 📄 License
MIT License









