# 🍽️ Food Freshness Checker

A web-based machine learning application that analyzes uploaded food images and predicts their freshness level to help users decide whether the food is safe to consume.

---

## ✨ Features
- Upload food images through a simple web interface
- Predicts food quality as **Fresh / Okay / Avoid**
- Displays uploaded image with prediction
- Clean and responsive UI
- Built with scalability in mind for advanced ML models

---

## 🛠 Tech Stack
- **Python**
- **Flask**
- **HTML, CSS**
- **Machine Learning (Image Classification)**
- **Git & GitHub**

---

## 📁 Project Structure
food_freshness_classifier/
│── app.py
│── model/
│ └── classifier.py
│── templates/
│ └── index.html
│── static/
│ └── style.css
│── requirements.txt
│── README.md


---

## ▶️ How to Run the Project Locally

```bash
git clone https://github.com/OMKARYEL/food_freshness_classifier.git
cd food_freshness_classifier
pip install -r requirements.txt
python app.py


Then open your browser and go to:
http://127.0.0.1:5000/




## 🚀 Future Deployment Plan

This project is designed to be easily deployable as a web application.

Planned deployment options include:

- **Render / Railway** for hosting the Flask web application
- **Hugging Face Spaces** for showcasing the ML model with a live demo
- **Docker (optional)** for containerized deployment
- **Cloud storage** for handling uploaded images securely

Once deployed, users will be able to access the Food Freshness Checker through a public URL without any local setup.
