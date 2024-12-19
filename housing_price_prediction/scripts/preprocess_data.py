import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load the dataset
    data_path = '/Users/suryanshu/Downloads/Housing.csv'
    housing_data = pd.read_csv(data_path)

    # One-hot encode categorical variables
    housing_data_encoded = pd.get_dummies(housing_data, drop_first=True)

    # Split the data into features and target variable
    X = housing_data_encoded.drop('price', axis=1)
    y = housing_data_encoded['price']

    # Split the dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f'Training data shape: {X_train.shape}')
    print(f'Testing data shape: {X_test.shape}')