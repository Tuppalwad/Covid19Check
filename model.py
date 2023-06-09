from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def check_is_covid(symptoms):
    data = pd.read_csv('Cleaned-Data.csv')

    int64_cols = data.select_dtypes(include='int64').columns

    # Transform columns with int64 dtype to boolean
    data[int64_cols] = data[int64_cols].astype(bool)

    # Replace '?' with NaN
    data = data.replace('?', np.NaN)

    # Preprocess the data
    X = data.drop(['Country', 'Severity_Mild', 'Severity_Moderate', 'Severity_None', 'Severity_Severe'], axis=1)
    y = data[['Severity_Mild', 'Severity_Moderate', 'Severity_None', 'Severity_Severe']].idxmax(axis=1)

    # Encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the model
    model = RandomForestClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Prepare the input data
    input_data = pd.DataFrame([symptoms], columns=X.columns)

    # Handle NaN and infinity values
    input_data = input_data.fillna(0)
    input_data = input_data.replace([np.inf, -np.inf], np.finfo(np.float32).max)

    # Make sure the input data has the same columns as the training data
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = le.inverse_transform(prediction)

    # Determine if the person has COVID-19
    has_covid = 'Severity_Severe' in predicted_class

    return has_covid


# symptoms = {
#     'Fever': 1,
#     'Cough': 1,
#     # 'Shortness_of_breath': 1,
#     # 'Fatigue': 1,
#     # 'Sore_throat': 1,
#     'Headache': 1,
#     'Body_aches': 1,
#     'Loss_of_taste_smell': 1,
#     'Congestion': 1,
#     'Nausea': 1,
#     'Diarrhea': 1
# }

# result = check_is_covid(symptoms)
# print(result)
