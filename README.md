# 🏠 House Price Prediction System

A Machine Learning project that predicts California district-level median house values using **Linear Regression** and an interactive **Streamlit Web Application**.

## 🚀 Project Overview

The House Price Prediction System uses the California Housing Dataset to estimate median house values based on different housing and geographical features.

The project includes data analysis, model training, model evaluation, and a web-based prediction interface.

## ✨ Features

- Exploratory Data Analysis (EDA)
- Data visualization using Matplotlib and Seaborn
- Linear Regression Machine Learning model
- Model evaluation using MAE, MSE, RMSE and R² Score
- Interactive Streamlit web application
- Real-time predictions from user inputs
- Model serialization using Joblib

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Streamlit | Web application |
| Joblib | Model saving and loading |

## 📊 Dataset Information

**Dataset:** California Housing Dataset

The dataset contains the following features:

- MedInc – Median income
- HouseAge – House age
- AveRooms – Average rooms
- AveBedrms – Average bedrooms
- Population – Population
- AveOccup – Average occupancy
- Latitude – Geographical latitude
- Longitude – Geographical longitude

**Target Variable:** `MedHouseVal`

The target represents median house value in units of $100,000.

## 📁 Project Structure

```text
House-Price-Prediction-System/
│
├── app.py
├── train_model.py
├── house_price_model.joblib
├── requirements.txt
├── README.md
└── house_price_prediction_system.ipynb
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/House-Price-Prediction-System.git
```

### 2. Open project folder

```bash
cd House-Price-Prediction-System
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning model

```bash
python train_model.py
```

This generates the trained model file:

```text
house_price_model.joblib
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

## 📈 Model Evaluation

The Linear Regression model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Actual evaluation results are printed when you run `train_model.py`.

## 🌐 Deployment

This project can be deployed using **Streamlit Community Cloud**.

1. Upload the project to GitHub.
2. Visit https://share.streamlit.io/
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Click Deploy.

Make sure the trained `house_price_model.joblib` file is included in your repository.

## 🎯 Future Improvements

- Compare Linear Regression with Random Forest and Decision Tree.
- Improve model accuracy.
- Add actual vs predicted visualizations.
- Add feature importance analysis.
- Improve the Streamlit dashboard UI.
- Implement advanced Machine Learning models.

## 👨‍💻 Author

**Sultan**

GitHub: https://github.com/sultancodesss

LinkedIn: https://www.linkedin.com/in/sultan-codes/

---

⭐ If you find this project useful, consider giving it a star on GitHub!

**Note:** This is an educational Machine Learning project. Predictions represent district-level median house values, not individual property valuations.
