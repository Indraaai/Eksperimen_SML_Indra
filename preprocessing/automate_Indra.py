

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Load dataset dari file CSV
    
    Parameters:
    -----------
    file_path : str
        Path ke file CSV
        
    Returns:
    --------
    df : DataFrame
        Dataset yang telah dimuat
    """
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    return df


def remove_duplicates(df):
    """
    Menghapus data duplikat
    
    Parameters:
    -----------
    df : DataFrame
        Dataset asli
        
    Returns:
    --------
    df_clean : DataFrame
        Dataset tanpa duplikat
    """
    before = df.shape[0]
    df_clean = df.drop_duplicates()
    after = df_clean.shape[0]
    print(f"Duplicates removed: {before - after}")
    return df_clean


def encode_categorical(df):
    """
    Encoding kolom kategorikal (gender dan smoking_history)
    
    Parameters:
    -----------
    df : DataFrame
        Dataset dengan kolom kategorikal
        
    Returns:
    --------
    df : DataFrame
        Dataset dengan kolom yang sudah di-encode
    le_gender : LabelEncoder
        Encoder untuk gender
    le_smoking : LabelEncoder
        Encoder untuk smoking_history
    """
    df = df.copy()
    
    le_gender = LabelEncoder()
    le_smoking = LabelEncoder()
    
    df['gender'] = le_gender.fit_transform(df['gender'])
    df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])
    
    print("Categorical encoding completed")
    print(f"Gender encoding: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")
    print(f"Smoking encoding: {dict(zip(le_smoking.classes_, le_smoking.transform(le_smoking.classes_)))}")
    
    return df, le_gender, le_smoking


def split_features_target(df, target_col='diabetes'):
    """
    Memisahkan fitur dan target variable
    
    Parameters:
    -----------
    df : DataFrame
        Dataset lengkap
    target_col : str
        Nama kolom target
        
    Returns:
    --------
    X : DataFrame
        Fitur
    y : Series
        Target variable
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data menjadi training dan testing set
    
    Parameters:
    -----------
    X : DataFrame
        Fitur
    y : Series
        Target variable
    test_size : float
        Proporsi data test (default: 0.2)
    random_state : int
        Random seed (default: 42)
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standarisasi fitur numerik
    
    Parameters:
    -----------
    X_train : array-like
        Data training
    X_test : array-like
        Data testing
        
    Returns:
    --------
    X_train_scaled : array
        Data training yang telah di-scale
    X_test_scaled : array
        Data testing yang telah di-scale
    scaler : StandardScaler
        Scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Feature scaling completed")
    print(f"Mean: {X_train_scaled.mean(axis=0).round(2)}")
    print(f"Std: {X_train_scaled.std(axis=0).round(2)}")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(file_path, test_size=0.2, random_state=42, save_clean=False):
    """
    Fungsi utama untuk melakukan preprocessing secara otomatis
    
    Parameters:
    -----------
    file_path : str
        Path ke file CSV
    test_size : float
        Proporsi data test (default: 0.2)
    random_state : int
        Random seed (default: 42)
    save_clean : bool
        Simpan data clean ke CSV (default: False)
        
    Returns:
    --------
    X_train_scaled : array
        Data training yang siap dilatih
    X_test_scaled : array
        Data testing yang siap dilatih
    y_train : Series
        Target training
    y_test : Series
        Target testing
    scaler : StandardScaler
        Scaler object
    encoders : dict
        Dictionary berisi encoder untuk gender dan smoking_history
    """
    print("="*60)
    print("Starting Automated Preprocessing Pipeline")
    print("="*60)
    
    # 1. Load data
    print("\n[Step 1/6] Loading data...")
    df = load_data(file_path)
    
    # 2. Remove duplicates
    print("\n[Step 2/6] Removing duplicates...")
    df_clean = remove_duplicates(df)
    
    # 3. Encode categorical
    print("\n[Step 3/6] Encoding categorical features...")
    df_clean, le_gender, le_smoking = encode_categorical(df_clean)
    
    # Save clean data if requested
    if save_clean:
        clean_path = 'data_clean.csv'
        df_clean.to_csv(clean_path, index=False)
        print(f"\nClean data saved to: {clean_path}")
    
    # 4. Split features and target
    print("\n[Step 4/6] Splitting features and target...")
    X, y = split_features_target(df_clean)
    
    # 5. Split train and test
    print("\n[Step 5/6] Splitting train and test...")
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
    
    # 6. Scale features
    print("\n[Step 6/6] Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Print summary
    print("\n" + "="*60)
    print("Preprocessing Completed Successfully!")
    print("="*60)
    print(f"Final training data shape: {X_train_scaled.shape}")
    print(f"Final testing data shape: {X_test_scaled.shape}")
    print(f"Target distribution:")
    print(y_train.value_counts())
    print(f"Target percentage:")
    print(y_train.value_counts(normalize=True) * 100)
    
    # Store encoders in dictionary
    encoders = {
        'gender': le_gender,
        'smoking_history': le_smoking
    }
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders


if __name__ == "__main__":
    # Contoh penggunaan
    file_path = '../diabetes_prediction_dataset_raw/data.csv'
    
    # Jalankan preprocessing
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data(
        file_path=file_path,
        test_size=0.2,
        random_state=42,
        save_clean=True
    )
    
    print("\n" + "="*60)
    print("Data siap untuk training model!")
    print("="*60)
