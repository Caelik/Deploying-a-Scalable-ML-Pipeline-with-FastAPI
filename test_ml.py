import pytest
import numpy as np
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score

# Sample data for testing
sample_data = {
    'age': [39, 50, 38, 53, 28, 37, 49, 52, 31, 42],
    'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private', 
                  'Self-emp-not-inc', 'Private', 'Self-emp-not-inc', 'Private', 'Private'],
    'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Masters', 
                  'Masters', '9th', 'HS-grad', 'Assoc-acdm', 'Bachelors'],
    'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 
                       'Married-civ-spouse', 'Married-civ-spouse', 'Married-civ-spouse',
                       'Married-spouse-absent', 'Never-married', 'Married-civ-spouse', 'Divorced'],
    'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 
                   'Prof-specialty', 'Sales', 'Craft-repair', 'Transport-moving', 
                   'Machine-op-inspct', 'Exec-managerial', 'Prof-specialty'],
    'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Wife', 
                     'Husband', 'Own-child', 'Not-in-family', 'Husband', 
                     'Own-child', 'Unmarried'],
    'race': ['White', 'Black', 'Asian-Pac-Islander', 'White', 'Black',
             'White', 'White', 'Black', 'White', 'Asian-Pac-Islander'],
    'sex': ['Male', 'Female', 'Male', 'Female', 'Male', 
             'Female', 'Male', 'Female', 'Male', 'Female'],
    'native-country': ['United-States', 'Cuba', 'Jamaica', 
                       'United-States', 'United-States', 'India', 
                       'United-States', 'Mexico', 'United-States', 'United-States'],
    'salary': ['<=50K', '<=50K', '>50K', '>50K', '<=50K',
               '>50K', '<=50K', '<=50K', '>50K', '<=50K']
}

# Convert sample data to DataFrame
data = pd.DataFrame(sample_data)

def test_train_model_type():
    """
    Test that the train_model function returns an instance of RandomForestClassifier.
    """
    X_train = pd.get_dummies(data.drop('salary', axis=1))
    y_train = data['salary']
    
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), 'The model should be of type RandomForestClassifier.'

def test_compute_model_metrics():
    """
    Test that the compute_model_metrics function returns expected precision, recall, and F1 score.
    """
    y_true = np.array(['>50K', '<=50K', '>50K', '>50K', '<=50K'])
    y_pred = np.array(['>50K', '<=50K', '>50K', '>50K', '>50K'])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred, pos_label='>50K')

    # Check expected values (you can replace these with the actual expected values for your test case)
    expected_precision = 0.75
    expected_recall = 1.0
    expected_fbeta = 0.8571428571428571

    assert precision == expected_precision, f'Expected precision {expected_precision}, got {precision}.'
    assert recall == expected_recall, f'Expected recall {expected_recall}, got {recall}.'
    assert fbeta == expected_fbeta, f'Expected F-beta {expected_fbeta}, got {fbeta}.'

def test_data_split_size():
    """
    Test that the train and test datasets have the expected size after splitting.
    """
    train_size = 0.8
    test_size = 0.2
    
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    
    assert train.shape[0] == int(train_size * data.shape[0]), 'Train dataset size is incorrect.'
    assert test.shape[0] == int(test_size * data.shape[0]), 'Test data size is incorrect.'

# Run tests
if __name__ == '__main__':
    pytest.main()
