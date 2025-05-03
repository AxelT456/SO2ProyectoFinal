import pandas as pd
import matplotlib.pyplot as plt
import os

# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)

# --- PRIMERA FUNCIÓN ---
# Comparación de tiempos por tipo de dato (Cannon vs Secuencial) para un tamaño dado
def graficar_tiempo_por_tipo(tamano):
    # Leer los archivos CSV
    df_cannon = pd.read_csv(f"csv/i5_10400_4.10GHz_12hilos/memoria/cannon_{tamano}.csv")
    df_secuencial = pd.read_csv(f"csv/i5_10400_4.10GHz_12hilos/memoria/secuencial_{tamano}.csv")

    tipos = df_cannon['Tipo']
    promedios_cannon = df_cannon[["T1", "T2", "T3", "T4", "T5"]].mean(axis=1)
    promedios_secuencial = df_secuencial[["T1", "T2", "T3", "T4", "T5"]].mean(axis=1)

    x = range(len(tipos))

    plt.figure(figsize=(10,6))
    plt.bar(x, promedios_cannon, width=0.4, label='Cannon', align='center')
    plt.bar([i + 0.4 for i in x], promedios_secuencial, width=0.4, label='Secuencial', align='center')
    plt.xticks([i + 0.2 for i in x], tipos)
    plt.xlabel('Tipo de dato')
    plt.ylabel('Memoria promedio (BM)')
    plt.title(f'Memoria promedio por tipo de dato - Matriz {tamano}x{tamano}\n Intel Core i5-10400 @ 4.10GHz - 12 hilos')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output/i5_10400_4.10GHz_12hilos/memoria/memoria_por_tipo_{tamano}.jpg")

    plt.show()
    plt.close()

if __name__ == "__main__":
    graficar_tiempo_por_tipo(2048)  # o el tamaño que quieras probar


