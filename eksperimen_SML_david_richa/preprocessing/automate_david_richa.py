from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd

# menerima dataset yang sudah di load sebelumnya


def preprocess(data, target_col, file_path, threshold=6.5):
    # Detect duplicate rows
    duplicate_count = data.duplicated().sum()
    print(f"Jumlah duplikat ditemukan: {duplicate_count}")

    # Remove duplicates
    data = data.drop_duplicates()
    print(f"Ukuran data setelah hapus duplikat: {data.shape}")

    # encoding quality
    data[target_col] = (data[target_col] > threshold).astype(int)
    print(f"kolom", [target_col], "encoding quality sedang diproses")

    num_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print("kolom numerik terdeteksi")

    # Mendapatkan nama kolom tanpa kolom target
    column_names = data.columns.drop(target_col)

    # membuat DF kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)

    # menyimpan kolom sebagai header tanpa data
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    # Pastikan target_column tidak ada di numeric_features
    # atau categorical_features
    if target_col in num_features:
        num_features.remove(target_col)

    num_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features)
        ]
    )

    # Memisahkan target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Membagi data
    X_scaled = preprocessor.fit_transform(X)
    df_preprocessed = pd.DataFrame(X_scaled, columns=num_features)
    df_preprocessed[target_col] = y.values

    df_preprocessed.to_csv(file_path, index=False)
    print(f"Hasil preprocessing berhasil disimpan ke: {file_path}")

    return df_preprocessed


if __name__ == "__main__":
    print("memulai preprocessing")
    data = pd.read_csv('../winequality-red_raw.csv')
    ready_data = preprocess(data,
                            "quality",
                            'winequality-red_preprocessed.csv')
    print("preprocessing selesai")
