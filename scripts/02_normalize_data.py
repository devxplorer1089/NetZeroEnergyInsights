import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def normalize_data(file_path, output_path):
    """
    Normalizes the dataset using MinMax scaling for numerical features and encodes categorical features.
    """
    df = pd.read_csv(file_path, low_memory=False)
    
    # Separate numerical and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Handle NaNs before scaling
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Scale numerical features
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    df.to_csv(output_path, index=False)
    print(f"Normalized dataset saved to {output_path}")
    
# Example usage
if __name__ == "__main__":
    normalize_data("data/cleaned_data.csv", "data/normalized_data.csv")
