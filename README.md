Insurance Risk Prediction System
Project Description

The Insurance Risk Prediction System is a machine learning-powered decision support application developed to assist insurance companies in evaluating the risk level of policy applicants. The system leverages an Extreme Gradient Boosting (XGBoost) classification model to analyse customer demographic, health, financial, and policy-related information and predict the applicant's insurance risk category.

The application automates the underwriting process by generating a predicted Risk Category (Low, Medium, High, or Critical), along with the model's confidence score, probability distribution across all risk classes, key risk factors influencing the prediction, and underwriting recommendations.

To ensure accurate predictions, the system incorporates a comprehensive data preprocessing pipeline, including binary encoding, ordinal encoding, one-hot encoding, feature engineering, logarithmic transformation, and feature scaling. These preprocessing steps mirror those used during model training, ensuring consistency between training and prediction.

The interactive user interface, developed using Gradio, enables users to input applicant information through an intuitive dashboard and instantly receive predictive insights. This allows insurance professionals to make faster, more informed underwriting decisions while improving risk assessment and operational efficiency.

Key Features
Predicts insurance risk category using an XGBoost classification model.
Automatically performs feature engineering and data preprocessing.
Calculates applicant risk level with confidence scores.
Displays class probabilities for all risk categories.
Identifies major factors contributing to the prediction.
Provides underwriting recommendations based on predicted risk.
Interactive and user-friendly web interface developed with Gradio.
Ready for deployment on cloud platforms such as Hugging Face Spaces or Streamlit Cloud.
Technologies Used
Python
XGBoost
Scikit-learn
Pandas
NumPy
Joblib
Gradio
Project Objective

The primary objective of this project is to develop an intelligent insurance risk prediction system capable of supporting underwriting decisions through machine learning techniques. By accurately classifying applicants into different risk categories, the system helps insurers improve decision-making, reduce financial losses, enhance operational efficiency, and deliver more personalised insurance services.
