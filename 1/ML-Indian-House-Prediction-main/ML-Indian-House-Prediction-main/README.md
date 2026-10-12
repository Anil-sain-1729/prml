# India House Price Prediction Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project focuses on predicting house prices in India using machine learning techniques. The model was trained on a subset of the [India Housing Prices Dataset](https://www.kaggle.com/datasets/ankushpanday1/india-house-price-prediction/data) from Kaggle.  This dataset provides detailed insights into housing market trends across various Indian states, including property types, pricing, location, and amenities.

**Key Features:**

* **Data Exploration and Preprocessing:**  Includes exploratory data analysis (EDA) to understand the data and preprocessing steps to prepare it for modeling.
* **Model Training:**  Employs various regression models including Linear Regression, Decision Tree, Random Forest, and XGBoost.
* **Hyperparameter Tuning:**  Parameter tuning was performed for all models, with a focus on Grid Search for optimizing XGBoost.
* **Performance Evaluation:**  Comprehensive evaluation of each model using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).

## Dataset

The model was trained using the [India Housing Prices Dataset](https://www.kaggle.com/datasets/ankushpanday1/india-house-price-prediction/data) available on Kaggle.

> The India Housing Prices Dataset contains 2.5 lakh rows and 23 columns, offering detailed insights into housing market trends across various Indian states. It includes attributes related to property types, pricing, location, and amenities, making it suitable for analysis ranging from basic descriptive statistics to advanced machine learning applications.

**Note:** Due to computational limitations, this project utilized a sample of approximately 100,000 rows from the original dataset.

## Business Applications

This house price prediction model can be valuable for:

- 🏘️ **Real Estate Agents**: Quickly estimate property values based on location and features
- 💰 **Property Investors**: Make data-driven decisions for investment opportunities across Indian states
- 🏦 **Banks & Lenders**: Assist in property valuation for mortgage and loan assessments
- 🏗️ **Property Developers**: Analyze market trends and optimize pricing strategies for new developments
- 🏡 **Home Buyers**: Get fair price estimates before making purchase decisions
- 📊 **Market Analysts**: Study housing market trends and price patterns across different Indian regions

The model's high accuracy (99.6% R² score with XGBoost) makes it a reliable tool for stakeholders in the Indian real estate market to make informed decisions.

## Methodology

The following steps were undertaken in this project:

1. **Exploratory Data Analysis (EDA):**  Initial exploration of the dataset to understand its structure, identify patterns, and gain insights into the features.
2. **Data Preprocessing:**
    * **Dropping Unnecessary Columns:** Identification and removal of irrelevant or redundant columns.
    * **Ordinal Encoding:** Encoding of ordinal categorical features into numerical representations.
    * **Data Sampling:**  Selection of a subset of approximately 100,000 rows from the dataset due to limited computational resources.
    * **One-Hot Encoding:** Application of One-Hot Encoding using `DictVectorizer` to convert remaining categorical features into a numerical format suitable for machine learning models.
3. **Model Training and Evaluation:**
    * Training of the following regression models:
        * Linear Regression
        * Decision Tree
        * Random Forest
        * XGBoost
    * Parameter tuning for each model to optimize performance.
    * Grid Search was specifically used to fine-tune the hyperparameters of the XGBoost model.
4. **Performance Evaluation:**  Evaluation of each trained model using the following metrics:
    * **Mean Absolute Error (MAE):** Average absolute difference between the predicted and actual prices.
    * **Mean Squared Error (MSE):** Average of the squared differences between the predicted and actual prices.
    * **Root Mean Squared Error (RMSE):** Square root of the MSE, providing an error metric in the original unit of the target variable.
    * **R-squared (R2):**  A statistical measure representing the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Results

The following performance metrics were achieved by the trained models:

| Model | MAE | MSE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| Linear Regression | 81.75 | 10364.25 | 101.80 | 0.488 |
| Decision Tree | 7.70 | 96.81 | 9.84 | 0.995 |
| Random Forest | 50.60 | 3910.55 | 62.53 | 0.807 |
| XGBoost | 6.96 | 78.74 | 8.87 | 0.996 |

### Bar Plot Comparison of Models

![Model's Result Using Bar Plot](https://github.com/izaanz/ML-Indian-House-Prediction/blob/main/img/model_comparison_bar_chart.png)

### Radar Chart Models Comparison
![Model's Result Using Radar](https://github.com/izaanz/ML-Indian-House-Prediction/blob/main/img/model_comparison_radar_chart.png)

## How to Use the Model

### Prerequisites

- **Pipenv:** For managing Python environments and dependencies.
- **Docker:** For containerized deployment of the model.
- **Flask:** For app/webservice.

### Running Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/izaanz/ML-Indian-House-Prediction
   navigate to the cloned directory
   ```

2. **Setup Environment:**
   ```bash
   pipenv install
   pipenv shell
   ```

3. **Run the Model:**
   
   Note: You may have to run `python train.py` if the model_.bin doesn't validate on your end.
   
   ```bash
   python predict.py
   ```
   This will start a local server where you can send requests to get predictions.

4. **Testing Predictions:**

   You can use predict_test.py to test the predictions.
   
   ```bash
   python predict_test.py
   ```
   This will return a predicted house priced based on the stored data in the file.

### Docker Deployment

To deploy using Docker:

1. **Build the Docker Image:**
   ```bash
   docker build -t house-price-predictor .
   ```

2. **Run the Docker Container:**
   ```bash
   docker run -p 9696:9696 house-price-predictor
   ```

   The model will be accessible at `http://localhost:9696/predict`.

This Dockerfile sets up a Python 3.11 environment, installs Pipenv, and copies the required files into the container. It then exposes port 9696 and sets up the Waitress server to serve the model.


### Interacting with the Model

When testing the model, you can use the following JSON structure for a student's data:


```json
{
'State': 'telangana',
 'City': 'warangal',
 'Property_Type': 0.0,
 'BHK': 1,
 'Size_in_SqFt': 2059,
 'Price_per_SqFt': 0.24,
 'Year_Built': 1995,
 'Furnished_Status': 2.0,
 'Floor_No': 0,
 'Total_Floors': 26,
 'Age_of_Property': 30,
 'Nearby_Schools': 7,
 'Nearby_Hospitals': 6,
 'Public_Transport_Accessibility': 0.0,
 'Parking_Space': 'no',
 'Security': 0.0,
 'Amenities': 'garden, pool, gym, playground, clubhouse',
 'Facing': 2.0,
 'Owner_Type': 'broker',
 'Availability_Status': 'under_construction'
}
```
Use predict_test.py to send test queries to your model:

- Use `predict_test.py` to send test queries to your model:
  ```bash
  python predict_test.py
  ```

  Modify this script to format your input data as per the model's expectations.

## Deployed on Cloud

This is a machine learning-based web application for predicting house prices in India. The app is powered by a regression model that takes various house attributes (such as location, size, number of rooms, etc.) to predict the price of a house.

The app is deployed on the cloud and can be accessed through the following link:

https://ml-indian-house-prediction.onrender.com/

![Cloud Deployment](https://github.com/izaanz/ML-Indian-House-Prediction/blob/main/img/frontend.png)

You can input house details into the app, and it will generate a predicted price based on the trained model.

## Contributions

Contributions to improve the model, enhance feature sets, or optimize the deployment process are welcome. Please submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Special thanks to Kaggle for providing the dataset.

---

This README provides a comprehensive guide to understanding, using, and deploying the Depression Prediction Model. For any issues or further information, feel free to open an issue in this repository.
