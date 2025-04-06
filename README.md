# Heart-Disease-Data-Analysis

This project focuses on analyzing and predicting the likelihood of heart disease using a structured dataset. It uses data preprocessing, visualization, and machine learning models to understand the impact of various health and demographic factors on heart disease.

## 📁 Dataset

The dataset used in this project is `Heart_Disease.csv`, which contains the following key features:

- `sex`: Gender of the individual  
- `age`: Age in years  
- `education`: Level of education  
- `smokingStatus`: Whether the individual is a smoker  
- `cigsPerDay`: Number of cigarettes smoked per day  
- `BPMeds`: Whether the person is on blood pressure medication  
- `prevalentStroke`: History of stroke  
- `prevalentHyp`: Prevalent hypertension  
- `diabetes`: Diabetes status  
- `totChol`: Total cholesterol level  
- `sysBP`: Systolic blood pressure  
- `diaBP`: Diastolic blood pressure  
- `BMI`: Body Mass Index  
- `heartRate`: Heart rate  
- `glucose`: Glucose level  
- `TenYearCHD`: Target variable — indicates risk of coronary heart disease in 10 years

## 🧠 Project Goals

- Perform **exploratory data analysis (EDA)** to uncover patterns and relationships.
- Preprocess and clean the dataset by handling missing values.
- Visualize key health indicators using libraries like Matplotlib and Seaborn.
- Build and evaluate multiple **machine learning models** for prediction:
  - Logistic Regression
  - Random Forest
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
- Measure performance using accuracy, precision, recall, F1 score, and confusion matrix.

## 🛠️ Tools & Technologies

- Python  
- Pandas, NumPy  
- Seaborn, Matplotlib  
- Scikit-learn (sklearn)  
- Jupyter Notebook  

## 📊 Visualizations

The notebook includes insightful visualizations to better understand:
- Gender-wise and age-wise distribution of heart disease
- Correlation heatmaps
- Distributions of BMI, cholesterol, glucose, etc.
- Model performance comparisons

## ✅ Results

- The models were trained and evaluated, with the best-performing model achieving high accuracy in identifying individuals at risk.
- Logistic Regression and Random Forest showed promising performance in terms of precision and recall.

## 📁 File Structure

```
📦Heart Disease Prediction
 ┣ 📄Heart_Disease.csv
 ┣ 📄Heart Disease Dataset.ipynb
 ┗ 📄README.md
```

## 🚀 How to Run

1. Clone this repository.
2. Install required dependencies:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open the notebook `Heart Disease.ipynb` in Jupyter.
4. Run all cells to view EDA, model building, and predictions.

## 📌 Conclusion

This project demonstrates how machine learning can be applied to real-world health data to help predict heart disease risk and potentially aid early diagnosis and prevention.
