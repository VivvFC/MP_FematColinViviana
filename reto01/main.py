import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_text

np.random.seed(42)

COLORES_ESPECIES = {
    'Adelie': '#FF6B6B',
    'Chinstrap': '#4ECDC4',
    'Gentoo': '#45B7D1'
}

def winsorizacion(df, columnas):
    df_w = df.copy()
    for especie in df_w["species"].unique():
        subset = df_w["species"] == especie
        for col in columnas:
            Q1 = df_w.loc[subset, col].quantile(0.25)
            Q3 = df_w.loc[subset, col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_w.loc[subset, col] = df_w.loc[subset, col].clip(lower, upper)
    return df_w

def analizarDesbalance(df, columna):
    conteo = df[columna].value_counts()
    proporciones = df[columna].value_counts(normalize=True)
    print("\nConteo de clases:\n", conteo)
    print("\nProporciones:\n", proporciones.round(3))
    ratio = conteo.max() / conteo.min()
    print("\nRatio desbalance:", round(ratio,2))
    if ratio < 1.5:
        print("Dataset balanceado")
    elif ratio < 3:
        print("Desbalance leve")
    else:
        print("Desbalance fuerte")

def clasificadorHumano(row):
    if row["flipper_length_mm"] > 205 and row["bill_depth_mm"] <= 17.65:
        return "Gentoo"
    if row["bill_length_mm"] <= 42.35:
        return "Adelie"
    if row["bill_length_mm"] > 43.35 and row["flipper_length_mm"] <= 207:
        return "Chinstrap"
    return "Adelie"

def procesar_datos(ruta_csv):
    df_original = pd.read_csv(ruta_csv)
    df = df_original.dropna().reset_index(drop=True)

    print("📡 Transmisión recibida desde Estación Palmer...")
    print("")
    print(f"   📊 Registros de pingüinos cargados: {len(df)}")
    print(f"   🏝️  Islas monitoreadas: {df['island'].nunique()}")
    print(f"   🐧 Especies identificadas: {df['species'].nunique()}")
    print("")
    print("   Distribución por especie:")
    for species in df['species'].unique():
        count = (df['species'] == species).sum()
        print(f"      • {species}: {count} individuos")

    print("")
    print("✅ Datos listos para análisis")

    print("🔍 Primeros 10 registros del campo:")
    print("="*80)
    print(df.head(10))

    print("Dimensiones del dataset:\n")
    print(df.shape)
    print("\nColumnas del dataset:\n")
    print(df.columns)
    print("\nInformación básica del dataset:\n")
    df.info()

    print("\nDescripción estadística del dataset:")
    print(df.describe())

    print(df["species"].value_counts())
    print(df["island"].value_counts())
    print(df["sex"].value_counts())

    corr = df.corr(numeric_only=True)

    print("Promedios por especie:")
    print(df.groupby("species").mean(numeric_only=True))

    print("Promedios por isla: ")
    print(df.groupby("island").mean(numeric_only=True))

    print(pd.crosstab(df["species"], df["sex"]))

    numericas = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    df = winsorizacion(df, numericas)

    analizarDesbalance(df, "species")

    X = df[numericas]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 42,
        stratify = y
    )

    print("\nTrain size:", len(X_train))
    print("Test size:", len(X_test))

    print("\nDistribución en train:")
    print(y_train.value_counts(normalize=True))

    print("\nDistribución en test:")
    print(y_test.value_counts(normalize=True))

    predHumano = X_test.apply(clasificadorHumano, axis=1)
    accHumano = accuracy_score(y_test, predHumano)

    print("\nAccuracy humano:", accHumano)

    print("\nReporte humano:\n")
    print(classification_report(y_test, predHumano))

    print("\nMatriz de confusión humano:\n")
    print(confusion_matrix(y_test, predHumano))

    modelo = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    modelo.fit(X_train, y_train)
    predML = modelo.predict(X_test)

    accML = accuracy_score(y_test, predML)

    print("\nAccuracy ML:", accML)

    print("\nReporte ML:\n")
    print(classification_report(y_test, predML))

    print("\nMatriz de confusión ML:\n")
    print(confusion_matrix(y_test, predML))

    print("Resultado final")
    print("Accuracy humano:", accHumano)
    print("Accuracy ML:", accML)

    if accML > accHumano:
        print("\nTiene una mayor precisión la máquina.")
    else:
        print("\nTiene una mayor precisión el humano.")

    errores = X_test.copy()

    errores["real"] = y_test
    errores["humano"] = predHumano
    errores["ml"] = predML
    errores["errorHumano"] = errores["real"] != errores["humano"]
    errores["errorML"] = errores["real"] != errores["ml"]

    print("\nErrores humano:\n")
    print(errores[errores["errorHumano"]])

    print("\nErrores ML:\n")
    print(errores[errores["errorML"]])

    resultados_csv = pd.DataFrame({
        "predML": predML,
        "predHumano": predHumano
    })
    archivo_salida = "resultados_humano_vs_maquina.csv"
    resultados_csv.to_csv(archivo_salida, index=False)
    print(f"\nResultados exportados exitosamente a '{archivo_salida}'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        archivo_entrada = sys.argv[1]
        procesar_datos(archivo_entrada)
    else:
        print("Error: No se proporcionó el archivo CSV.")
        print("Uso correcto: python main.py ArchivoCSV.csv")