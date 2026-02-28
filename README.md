# AI-Powered Smart Energy Management System

## Description
Intelligent energy management system using **Machine Learning (XGBoost)** and **Deep Learning (LSTM)** to monitor, predict, and optimize energy consumption with a real-time dashboard.

---

## Key Features
- Real-time monitoring of energy consumption  
- Zone management and interactive dashboard  
- Alerts for anomalies and peak loads  
- Advanced analytics and manual scenario testing  

---

## Installation

bash
# 1. Clone the repository
git clone https://github.com/oumaima2024/smart-energy-management-ai.git
cd smart-energy-management-ai

# 2. Create virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run model fixer script
python Fixmodel.py

# 5. Start backend
python app.py

# 6. Start dashboard (in another terminal)

streamlit run Dashboard.py
