<img width="1835" height="640" alt="Capture d&#39;écran 2026-02-28 055915" src="https://github.com/user-attachments/assets/86de35d5-09aa-461c-bb7b-05f6ca6187bc" />
<img width="1763" height="752" alt="Capture d&#39;écran 2026-02-28 055904" src="https://github.com/user-attachments/assets/0a1b7d9f-2fc2-4fbf-be89-c9b9b41bd9be" />
<img width="1876" height="828" alt="Capture d&#39;écran 2026-02-28 055852" src="https://github.com/user-attachments/assets/a2adbfb7-2ec3-4c42-9417-e06ea983a16d" />
<img width="1897" height="820" alt="Capture d&#39;écran 2026-02-28 055838" src="https://github.com/user-attachments/assets/85207491-ee11-4c71-958c-8a7c97d67d8e" />
<img width="1919" height="812" alt="Capture d&#39;écran 2026-02-28 055823" src="https://github.com/user-attachments/assets/8483ea44-5356-4026-930a-d9dd4c223fdc" />
<img width="1919" height="743" alt="Capture d&#39;écran 2026-02-28 055806" src="https://github.com/user-attachments/assets/640f28e3-07b4-4da0-96fd-550a67c8138a" />
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
cd frontend
streamlit run app.py
