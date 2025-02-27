import pandas as pd
import numpy as np

def clean_data(file_path, output_path):
    """
    Cleans the dataset by handling missing values, removing duplicates,
    and performing necessary transformations.
    """
    df = pd.read_csv(file_path, low_memory=False)
    
    # Drop duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Fill missing numeric values with the median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill missing categorical values with the mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Standardizing column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")
    
if __name__ == "__main__":
    clean_data("data/original.csv", "data/cleaned_data.csv")