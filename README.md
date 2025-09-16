# 🛒 E-Commerce Customer Satisfaction Score Prediction (Deep Learning Model)

  ![image](https://github.com/VargheseTito/E-Commerce-Customer-Satisfaction-Score-Prediction-DL-Model/assets/110298267/e71b7160-24b3-4c2c-a59b-12b7bceed09b)

## 📌 Project Summary

### 🔍 Overview

This project focuses on predicting **Customer Satisfaction (CSAT) scores** using **Deep Learning (Artificial Neural Networks - ANN)**. In the e-commerce industry, understanding customer satisfaction through interactions and feedback is crucial for:

* Improving service quality
* Boosting customer retention
* Driving business growth

By leveraging a neural network model, this project aims to forecast CSAT scores accurately from customer interaction data, offering **real-time insights** that businesses can act upon.

---

### 📖 Project Background

Customer satisfaction is a key metric that drives **loyalty, repeat purchases, and referrals**. Traditionally, satisfaction is measured via surveys, but they:

* Take time to collect
* Capture only a portion of the customer experience

With deep learning, companies can now **predict satisfaction scores dynamically**, helping identify weak points and optimize service delivery **instantly**.

---

## 📂 Dataset Overview

The dataset contains **customer interaction records** from an e-commerce platform called **Shopzilla** (1 month).

**Features include:**

* **Unique ID** – identifier for each record
* **Channel Name** – service channel used by customer
* **Category / Sub-category** – type of issue raised
* **Customer Remarks** – feedback/comments
* **Order ID & Order Date** – order details
* **Issue Reported At / Responded At** – timestamps for response
* **Survey Response Date** – when CSAT survey was answered
* **Customer City** – city of the customer
* **Product Category** – product involved
* **Item Price** – purchase amount
* **Handling Time** – time taken by agent
* **Agent Name, Supervisor, Manager** – support staff info
* **Tenure Bucket** – agent’s experience level
* **Agent Shift** – shift timing (day/night)
* **CSAT Score (Target Variable)** – integer satisfaction score (0–4)

---

## 🎯 Project Goal

The primary objective is to **predict CSAT scores** using customer interaction data. This will enable:

* **Proactive service improvement**
* **Real-time monitoring of satisfaction**
* **Actionable insights** for management

---

## 🛠️ Tech Stack

* **Programming Language:** Python 🐍
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Handling & Analysis:** Pandas, NumPy, Scikit-learn
* **Model Saving/Loading:** Joblib, H5 Format
* **Frontend:** Streamlit
* **Visualization:** Matplotlib, Seaborn
* **Version Control:** Git & GitHub

---

## 🧠 Model Architecture

The model is built as a **multi-layer Artificial Neural Network (ANN)**:

* **Input Layer:** Takes in all selected features after preprocessing
* **Hidden Layers:**

  * Dense layers with **ReLU activation**
  * **Batch Normalization** for stable training
  * **Dropout layers** to prevent overfitting
* **Output Layer:**

  * Dense layer with **Softmax activation** for predicting CSAT scores (0–4)

The model is trained with **categorical crossentropy loss**, optimized using the **Adam optimizer**.

---

## 📊 Model Performance

From evaluation on the test dataset:

* **Overall Accuracy:** \~74%
* **Per-Class Performance:**

  * CSAT=0 → Precision 0.78, Recall 0.50
  * CSAT=1 → Precision 0.86, Recall 0.89
  * CSAT=2 → Precision 0.81, Recall 0.77
  * CSAT=3 → Precision 0.64, Recall 0.61
  * CSAT=4 → Precision 0.65, Recall 0.93

✅ **Insight:** The model performs well overall, especially in distinguishing satisfied (1, 2, 4) customers. Some classes like **3 (neutral)** are harder to classify due to overlapping feedback patterns.

---

## 🌐 Streamlit Frontend

The Streamlit app provides **two modes of prediction**:

1. **Sidebar Manual Input** → Enter interaction details manually and get an instant CSAT prediction.
2. **CSV Upload** → Upload a dataset of multiple records for **batch prediction**.

📸 **Frontend Preview:**
👉 *(Add screenshots of your Streamlit app here)*

---

## 📦 How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/CSAT-Prediction-DeepLearning.git
cd CSAT-Prediction-DeepLearning
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

* Use **transformers for text features** (customer remarks)
* Apply **hyperparameter tuning** for better performance
* Deploy as a **cloud-based service** (Azure/AWS/GCP)

---

✍️ **Author:** Souvik Karmakar
📌 *Full Stack Data Science & AI Certified | Data Analyst | Deep Learning Enthusiast*



