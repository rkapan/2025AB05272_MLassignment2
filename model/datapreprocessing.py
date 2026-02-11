

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def fetch_processed_data(path="data/bank-full.csv"):
    df = pd.read_csv(path, sep=';')

    # Encode target
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop('y', axis=1)
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test,scaler


# def fetch_processed_data_from_df(df):
#     # Encode target
#     df['y'] = df['y'].map({'yes': 1, 'no': 0})

#     # Encode categorical features
#     categorical_cols = df.select_dtypes(include=['object']).columns
#     le = LabelEncoder()
#     for col in categorical_cols:
#         df[col] = le.fit_transform(df[col])

#     X = df.drop('y', axis=1)
#     y = df['y']

#     return X, y 

def fetch_processed_data_from_df(df,scaler):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Only encode 'y' if it exists
    if 'y' in df.columns:
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
        categorical_cols = [col for col in categorical_cols if col != 'y']

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Ensure 'y' is present and not all NaN after mapping
    if 'y' in df.columns and df['y'].notna().all():
        print("Test data columns:", df.columns.tolist())
        print("Unique values in 'y':", df['y'].unique())
        
        X = df.drop('y', axis=1)
        y = df['y']
        #scaler = StandardScaler()
        X_scaled = scaler.transform(X)
        return X_scaled, y
    else:
        print("Else branch: 'y' column not found or contains only NaN in test data. Returning DataFrame and None.")
        return df, None
