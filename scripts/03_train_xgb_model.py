import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def train_xgb_model(file_path, output_model_path):
    """
    Trains an XGBoost regression model on the dataset and evaluates it using RMSE & R² score.
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Rename target column if needed
    if 'energy_consumption' not in df.columns and 'consommation_energie' in df.columns:
        df.rename(columns={'consommation_energie': 'energy_consumption'}, inplace=True)

    if 'energy_consumption' not in df.columns:
        raise KeyError("Target variable 'energy_consumption' not found in dataset.")

    # Apply log transformation to stabilize variance
    df['energy_consumption'] = np.log1p(df['energy_consumption'])

    # Drop non-numeric columns that are not useful for prediction
    irrelevant_cols = [
        'numero_dpe', 'nom_methode_dpe', 'version_methode_dpe',
        'nom_methode_etude_thermique', 'version_methode_etude_thermique',
        'date_visite_diagnostiqueur', 'date_etablissement_dpe', 'date_arrete_tarifs_energies',
        'commentaires_ameliorations_recommandations', 'explication_personnalisee',
        'commune', 'arrondissement', 'type_voie', 'nom_rue', 'numero_rue', 'batiment',
        'escalier', 'etage', 'porte', 'code_postal', 'code_insee_commune',
        'code_insee_commune_actualise', 'numero_lot', 'quote_part', 'partie_batiment',
        'organisme_certificateur', 'adresse_organisme_certificateur', 'date_reception_dpe',
        'geo_type', 'geo_adresse', 'geo_id', 'geo_l4', 'geo_l5'
    ]
    df = df.drop(columns=[col for col in irrelevant_cols if col in df.columns], errors='ignore')

    # Convert categorical columns into numeric values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Remove features with low variance
    df = df.loc[:, df.var() > 1e-5]

    # Scale features
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col != 'energy_consumption']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X = df.drop(columns=['energy_consumption'])
    y = df['energy_consumption']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    # Reverse log transformation
    y_test = np.expm1(y_test)
    y_pred = np.expm1(y_pred)

    # Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    model.save_model(output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    train_xgb_model("data/normalized_data.csv", "models/xgb_model_v2.json")

