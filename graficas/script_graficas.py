import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- FUNCIÓN GENERAL PARA SEPARAR POR TAMAÑO ---
def generar_df_tiempo(df_base, size):
    # Filtrar por tamaño
    df_filtrado = df_base[df_base['Size'] == size]
    # Crear tabla pivote para tener cada repetición en una columna
    df_pivot = df_filtrado.pivot_table(index="DataType", columns="Repetition", values="Time(s)") #"Time(s)" por "Time(ms)" en caso de ser necesario
    return df_pivot.reset_index()

def generar_df_memoria(df_base, size):
    # Filtrar por tamaño
    df_filtrado = df_base[df_base['Size'] == size]
    # Crear tabla pivote para tener cada repetición en una columna
    df_pivot = df_filtrado.pivot_table(index="DataType", columns="Repetition", values="Memory(MB)")
    return df_pivot.reset_index()

# --- FUNCIÓN GENERAL PARA SEPARAR POR TAMAÑO (CON THREAD PERCENTAGE) ---
def generar_df_tiempo_cuda(df_base, size):
    # Filtrar por tamaño
    df_filtrado = df_base[df_base['Size'] == size]
    # Crear tabla pivote para tener cada repetición en una columna
    df_pivot = df_filtrado.pivot_table(index=["DataType", "ThreadPercentage"], columns="Repetition", values="Time(s)")
    return df_pivot.reset_index()

def generar_df_memoria_cuda(df_base, size):
    # Filtrar por tamaño
    df_filtrado = df_base[df_base['Size'] == size]
    # Crear tabla pivote para tener cada repetición en una columna
    df_pivot = df_filtrado.pivot_table(index=["DataType", "ThreadPercentage"], columns="Repetition", values="Memory(MB)")
    return df_pivot.reset_index()


# --- FUNCIÓN PARA GRAFICAR CUDA VS OPENCL ---
def graficar_tiempo_comparativo(tamano):
    # Leer archivos base
    df_cuda = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/CUDA/cuda_results_IVAN.csv")
    df_opencl = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/OpenCL/opencl_resultsIvan.csv")
    
    # Generar DataFrames procesados
    df_cuda_res = generar_df_tiempo(df_cuda, size=tamano)
    df_opencl_res = generar_df_tiempo(df_opencl, size=tamano)
    
    # Calcular promedios
    promedios_cuda = df_cuda_res.iloc[:, 1:].mean(axis=1)
    promedios_opencl = df_opencl_res.iloc[:, 1:].mean(axis=1)
    tipos = df_cuda_res["DataType"]

    # Graficar
    plt.figure(figsize=(10,6))
    x = range(len(tipos))
    plt.bar(x, promedios_cuda, width=0.4, label='CUDA', align='center')
    plt.bar([i + 0.4 for i in x], promedios_opencl, width=0.4, label='OpenCL', align='center')
    plt.xticks([i + 0.2 for i in x], tipos, rotation=45)
    plt.xlabel('Tipo de dato')
    plt.ylabel('Tiempo promedio (ms)')
    plt.title(f'Tiempo promedio por tipo de dato - Matriz {tamano}x{tamano}\n i5_10400_4.10GHz')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()

    # --- FUNCIÓN PARA GRAFICAR CUDA VS OPENCL ---
def graficar_memoria_comparativo(tamano):
    # Leer archivos base
    df_cuda = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/CUDA/cuda_results_IVAN.csv")
    df_opencl = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/OpenCL/opencl_resultsIvan.csv")
    
    # Generar DataFrames procesados
    df_cuda_res = generar_df_memoria(df_cuda, size=tamano)
    df_opencl_res = generar_df_memoria(df_opencl, size=tamano)
    
    # Calcular promedios
    promedios_cuda = df_cuda_res.iloc[:, 1:].mean(axis=1)
    promedios_opencl = df_opencl_res.iloc[:, 1:].mean(axis=1)
    tipos = df_cuda_res["DataType"]

    # Graficar
    plt.figure(figsize=(10,6))
    x = range(len(tipos))
    plt.bar(x, promedios_cuda, width=0.4, label='CUDA', align='center')
    plt.bar([i + 0.4 for i in x], promedios_opencl, width=0.4, label='OpenCL', align='center')
    plt.xticks([i + 0.2 for i in x], tipos, rotation=45)
    plt.xlabel('Tipo de dato')
    plt.ylabel('Memoria promedio (MB)')
    plt.title(f'Memoria promedio por tipo de dato - Matriz {tamano}x{tamano}\n i5_10400_4.10GHz')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output/i5_10400_4.10GHz_12hilos/memoria_CUDAvsOpenCl/memoria_comparativo_{tamano}.jpg")
    plt.show()
    plt.close()

# --- FUNCIÓN PARA GRAFICAR SECUENCIAL VS OPENMP ---
def graficar_tiempo_sec_openmp(tamano):
    # Leer archivos base
    df_secuencial = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/Secuencial/secuencial_Ivan.csv")
    df_openmp = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/OMP/openMP_Ivan.csv")
    
    # Generar DataFrames procesados
    df_secuencial_res = generar_df_tiempo(df_secuencial, size=tamano)
    df_openmp_res = generar_df_tiempo(df_openmp, size=tamano)
    
    # Calcular promedios
    promedios_secuencial = df_secuencial_res.iloc[:, 1:].mean(axis=1)
    promedios_openmp = df_openmp_res.iloc[:, 1:].mean(axis=1)
    tipos = df_secuencial_res["DataType"]

    # Graficar
    plt.figure(figsize=(10,6))
    x = range(len(tipos))
    plt.bar(x, promedios_secuencial, width=0.4, label='Secuencial', align='center')
    plt.bar([i + 0.4 for i in x], promedios_openmp, width=0.4, label='OpenMP', align='center')
    plt.xticks([i + 0.2 for i in x], tipos, rotation=45)
    plt.xlabel('Tipo de dato')
    plt.ylabel('Tiempo promedio (s)')
    plt.title(f'Tiempo promedio por tipo de dato - Matriz {tamano}x{tamano}\n i5_10400_4.10GHz_12hilos')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output/i5_10400_4.10GHz_12hilos/CPU/tiempo_SecvsOMP/tiempo_sec_openmp_{tamano}.jpg")
    plt.show()
    plt.close()

def graficar_memoria_sec_openmp(tamano):
    # Leer archivos base
    df_secuencial = pd.read_csv("csv/intel core i7-10810u/Secuencial/resultados_secuencial_total.csv")
    df_openmp = pd.read_csv("csv/intel core i7-10810u/OMP/resultados_OMP.csv")
    
    # Generar DataFrames procesados
    df_secuencial_res = generar_df_memoria(df_secuencial, size=tamano)
    df_openmp_res = generar_df_memoria(df_openmp, size=tamano)
    
    # Calcular promedios
    promedios_secuencial = df_secuencial_res.iloc[:, 1:].mean(axis=1)
    promedios_openmp = df_openmp_res.iloc[:, 1:].mean(axis=1)
    tipos = df_secuencial_res["DataType"]

    # Graficar
    plt.figure(figsize=(10,6))
    x = range(len(tipos))
    plt.bar(x, promedios_secuencial, width=0.4, label='Secuencial', align='center')
    plt.bar([i + 0.4 for i in x], promedios_openmp, width=0.4, label='OpenMP', align='center')
    plt.xticks([i + 0.2 for i in x], tipos, rotation=45)
    plt.xlabel('Tipo de dato')
    plt.ylabel('Memoria promedio (MB)')
    plt.title(f'Memoria promedio por tipo de dato - Matriz {tamano}x{tamano}\n intel core i7-10810u')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output/intel core i7-10810u/CPU/memoria_SecvsOMP/memoria_sec_openmp_{tamano}.jpg")
    plt.show()
    plt.close()


def graficar_por_tipos_dato_por_grupo_tiempo():
    # Definir tamaños agrupados
    grupos_tamanos = {
        "Pequeños": [64, 128, 256, 512],
        "Medianos": [1024, 2048, 4096],
        "Grandes": [8192, 16384]
    }

    # Definir rutas para cada enfoque
    rutas = {
        "CUDA": "csv/i5_10400_4.10GHz_12hilos/CUDA/cuda_results_IVAN.csv",
        "OpenCL": "csv/i5_10400_4.10GHz_12hilos/OpenCL/opencl_resultsIvan.csv",
        "OpenMP": "csv/i5_10400_4.10GHz_12hilos/OMP/openMP_Ivan.csv",
        "Secuencial": "csv/i5_10400_4.10GHz_12hilos/Secuencial/secuencial_Ivan.csv"
    }

    # Colores para los tipos de dato
    colores = {
        "short": "#1f77b4",
        "long": "#ff7f0e",
        "float": "#2ca02c",
        "double": "#d62728"
    }

    for enfoque, ruta in rutas.items():
        # Leer datos
        df_base = pd.read_csv(ruta)
        tipos_dato = df_base["DataType"].unique()

        for grupo_nombre, tamanos in grupos_tamanos.items():
            # Crear figura
            plt.figure(figsize=(12, 8))
            bar_width = 0.2
            indice_barras = np.arange(len(tamanos))

            # Graficar cada tipo de dato con separación
            for i, tipo in enumerate(tipos_dato):
                promedios = []
                for size in tamanos:
                    df_filtrado = df_base[(df_base["Size"] == size) & (df_base["DataType"] == tipo)]
                    promedio = df_filtrado["Time(s)"].mean()
                    promedios.append(promedio)
                # Ajuste para que no se sobrepongan las barras
                posiciones = indice_barras + (i * bar_width)
                plt.bar(posiciones, promedios, width=bar_width, label=tipo, color=colores.get(tipo, "gray"))

            # Configurar gráfico
            plt.xlabel("Tamaño de matriz")
            plt.ylabel("Tiempo promedio (s)")
            plt.title(f"Tiempo promedio por tipo de dato - {enfoque} ({grupo_nombre})")
            plt.xticks(indice_barras + bar_width * (len(tipos_dato) - 1) / 2, tamanos)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            # Crear directorio si no existe
            output_dir = f"output/i5_10400_4.10GHz_12hilos/CPUvsGPU/tiempo/{enfoque}/{grupo_nombre}"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/promedios_tipos_dato_{grupo_nombre}.png")
            plt.show()
            plt.close()


def graficar_por_tipos_dato_por_grupo_memoria():
    # Definir tamaños agrupados
    grupos_tamanos = {
        "Pequeños": [64, 128, 256, 512],
        "Medianos": [1024, 2048, 4096],
        "Grandes": [8192, 16384]
    }

    # Definir rutas para cada enfoque
    rutas = {
        "CUDA": "csv/i5_10400_4.10GHz_12hilos/CUDA/cuda_results_IVAN.csv",
        "OpenCL": "csv/i5_10400_4.10GHz_12hilos/OpenCL/opencl_resultsIvan.csv",
        "OpenMP": "csv/i5_10400_4.10GHz_12hilos/OMP/openMP_Ivan.csv",
        "Secuencial": "csv/i5_10400_4.10GHz_12hilos/Secuencial/secuencial_Ivan.csv"
    }

    # Colores para los tipos de dato
    colores = {
        "short": "#1f77b4",
        "long": "#ff7f0e",
        "float": "#2ca02c",
        "double": "#d62728"
    }

    for enfoque, ruta in rutas.items():
        # Leer datos
        df_base = pd.read_csv(ruta)
        tipos_dato = df_base["DataType"].unique()

        for grupo_nombre, tamanos in grupos_tamanos.items():
            # Crear figura
            plt.figure(figsize=(12, 8))
            bar_width = 0.2
            indice_barras = np.arange(len(tamanos))

            # Graficar cada tipo de dato con separación
            for i, tipo in enumerate(tipos_dato):
                promedios = []
                for size in tamanos:
                    df_filtrado = df_base[(df_base["Size"] == size) & (df_base["DataType"] == tipo)]
                    promedio = df_filtrado["Memory(MB)"].mean()
                    promedios.append(promedio)
                # Ajuste para que no se sobrepongan las barras
                posiciones = indice_barras + (i * bar_width)
                plt.bar(posiciones, promedios, width=bar_width, label=tipo, color=colores.get(tipo, "gray"))

            # Configurar gráfico
            plt.xlabel("Tamaño de matriz")
            plt.ylabel("Memoria promedio (MB)")
            plt.title(f"Memoria promedio por tipo de dato - {enfoque} ({grupo_nombre})")
            plt.xticks(indice_barras + bar_width * (len(tipos_dato) - 1) / 2, tamanos)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            # Crear directorio si no existe
            output_dir = f"output/i5_10400_4.10GHz_12hilos/CPUvsGPU/memoria/{enfoque}/{grupo_nombre}"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/promedios_tipos_dato_{grupo_nombre}.png")
            plt.show()
            plt.close()


# --- FUNCIÓN PARA GRAFICAR CPU VS GPU POR TIPO DE DATO ---
def graficar_cpu_vs_gpu_por_tamaño_tiempo():
    # Definir tamaños clásicos divididos en pequeños, medianos y grandes
    tamanos = {
        "Pequeños": [64, 128, 256, 512],
        "Medianos": [1024, 2048, 4096],
        "Grandes": [8192, 16384]
    }
    
    # Definir rutas para cada enfoque
    rutas = {
        "CUDA": "csv/i5_10400_4.10GHz_12hilos/CUDA/cuda_results_IVAN.csv",
        "OpenCL": "csv/i5_10400_4.10GHz_12hilos/OpenCL/opencl_resultsIvan.csv",
        "OpenMP": "csv/i5_10400_4.10GHz_12hilos/OMP/openMP_Ivan.csv",
        "Secuencial": "csv/i5_10400_4.10GHz_12hilos/Secuencial/secuencial_Ivan.csv"
    }
    
    # Iterar por cada rango de tamaños
    for rango, tamaños in tamanos.items():
        # Leer todos los datos
        datos = {}
        for enfoque, ruta in rutas.items():
            df = pd.read_csv(ruta)
            tipos_dato = df["DataType"].unique()
            for tipo in tipos_dato:
                if tipo not in datos:
                    datos[tipo] = {"CUDA": [], "OpenCL": [], "OpenMP": [], "Secuencial": []}
                for tamaño in tamaños:
                    promedio = df[(df["Size"] == tamaño) & (df["DataType"] == tipo)]["Time(s)"].mean()
                    datos[tipo][enfoque].append(promedio)
        
        # Crear las gráficas para cada tipo de dato
        for tipo, resultados in datos.items():
            plt.figure(figsize=(12, 8))
            x = np.arange(len(tamaños))  # Posiciones en X para las barras
            width = 0.2  # Ancho de las barras
            
            # Añadir barras sin superposición
            plt.bar(x - width * 1.5, resultados["Secuencial"], width, label="Secuencial")
            plt.bar(x - width * 0.5, resultados["OpenMP"], width, label="OpenMP")
            plt.bar(x + width * 0.5, resultados["OpenCL"], width, label="OpenCL")
            plt.bar(x + width * 1.5, resultados["CUDA"], width, label="CUDA")
            
            # Configurar gráfico
            plt.xticks(x, tamaños)
            plt.xlabel("Tamaño de matriz")
            plt.ylabel("Tiempo promedio (s)")
            plt.title(f"Tiempo promedio para {tipo} - {rango}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            os.makedirs(f"output/i5_10400_4.10GHz_12hilos/CPU_vs_GPU/{rango}", exist_ok=True)
            plt.savefig(f"output/i5_10400_4.10GHz_12hilos/CPU_vs_GPU/{rango}/{tipo}_{rango}.png")
            plt.show()
            plt.close()

# --- FUNCIÓN PARA GRAFICAR CPU VS GPU POR TIPO DE DATO ---
def graficar_cpu_vs_gpu_por_tamaño_memoria():
    # Definir tamaños clásicos divididos en pequeños, medianos y grandes
    tamanos = {
        "Pequeños": [64, 128, 256, 512],
        "Medianos": [1024, 2048, 4096],
        "Grandes": [8192, 16384]
    }
    
    # Definir rutas para cada enfoque
    rutas = {
        "CUDA": "csv/i5_10400_4.10GHz_12hilos/CUDA/cuda_results_IVAN.csv",
        "OpenCL": "csv/i5_10400_4.10GHz_12hilos/OpenCL/opencl_resultsIvan.csv",
        "OpenMP": "csv/i5_10400_4.10GHz_12hilos/OMP/openMP_Ivan.csv",
        "Secuencial": "csv/i5_10400_4.10GHz_12hilos/Secuencial/secuencial_Ivan.csv"
    }
    
    # Iterar por cada rango de tamaños
    for rango, tamaños in tamanos.items():
        # Leer todos los datos
        datos = {}
        for enfoque, ruta in rutas.items():
            df = pd.read_csv(ruta)
            tipos_dato = df["DataType"].unique()
            for tipo in tipos_dato:
                if tipo not in datos:
                    datos[tipo] = {"CUDA": [], "OpenCL": [], "OpenMP": [], "Secuencial": []}
                for tamaño in tamaños:
                    promedio = df[(df["Size"] == tamaño) & (df["DataType"] == tipo)]["Memory(MB)"].mean()
                    datos[tipo][enfoque].append(promedio)
        
        # Crear las gráficas para cada tipo de dato
        for tipo, resultados in datos.items():
            plt.figure(figsize=(12, 8))
            x = np.arange(len(tamaños))  # Posiciones en X para las barras
            width = 0.2  # Ancho de las barras
            
            # Añadir barras sin superposición
            plt.bar(x - width * 1.5, resultados["Secuencial"], width, label="Secuencial")
            plt.bar(x - width * 0.5, resultados["OpenMP"], width, label="OpenMP")
            plt.bar(x + width * 0.5, resultados["OpenCL"], width, label="OpenCL")
            plt.bar(x + width * 1.5, resultados["CUDA"], width, label="CUDA")
            
            # Configurar gráfico
            plt.xticks(x, tamaños)
            plt.xlabel("Tamaño de matriz")
            plt.ylabel("Memoria promedio (MB)")
            plt.title(f"Memoria promedio para {tipo} - {rango}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            os.makedirs(f"output/i5_10400_4.10GHz_12hilos/CPU_vs_GPU/memoria/{rango}", exist_ok=True)
            plt.savefig(f"output/i5_10400_4.10GHz_12hilos/CPU_vs_GPU/memoria/{rango}/{tipo}_{rango}.png")
            plt.show()
            plt.close()


# --- FUNCIÓN PARA GRAFICAR CPUvsCPU ---
def graficar_memoria_comparativo_CPUvsCPU(tamano):
    # Leer archivos base
    df_ompIvan = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/OMP/openMP_Ivan.csv")
    df_secuencialIvan = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/Secuencial/secuencial_Ivan.csv")
    df_ompAlex = pd.read_csv("csv/intel core i7-10810u/OMP/resultados_OMP.csv")
    df_secuencialAlex = pd.read_csv("csv/intel core i7-10810u/Secuencial/resultados_secuencial_total.csv")
    
    # Generar DataFrames procesados
    df_ompIvan_res = generar_df_memoria(df_ompIvan, size=tamano)
    df_secIvan_res = generar_df_memoria(df_secuencialIvan, size=tamano)
    df_ompAlex_res = generar_df_memoria(df_ompAlex, size=tamano)
    df_secAlex_res = generar_df_memoria(df_secuencialAlex, size=tamano)
    
    # Calcular promedios
    promedios_ompIvan = df_ompIvan_res.iloc[:, 1:].mean(axis=1)
    promedios_secIvan = df_secIvan_res.iloc[:, 1:].mean(axis=1)
    promedios_ompAlex = df_ompAlex_res.iloc[:, 1:].mean(axis=1)
    promedios_secAlex = df_secAlex_res.iloc[:, 1:].mean(axis=1)
    tipos = df_ompIvan_res["DataType"]

    # Configuración de barras
    plt.figure(figsize=(12, 8))
    x = np.arange(len(tipos))
    width = 0.2  # Ancho de cada grupo de barras

    # Graficar cada grupo de barras
    plt.bar(x - 1.5 * width, promedios_ompIvan, width=width, label='OMP - i5_10400_4.10GHz_12hilos', align='center')
    plt.bar(x - 0.5 * width, promedios_secIvan, width=width, label='Secuencial - i5_10400_4.10GHz_12hilos', align='center')
    plt.bar(x + 0.5 * width, promedios_ompAlex, width=width, label='OMP - i7-10810u', align='center')
    plt.bar(x + 1.5 * width, promedios_secAlex, width=width, label='Secuencial - i7-10810u', align='center')

    # Personalizar ejes
    plt.xticks(x, tipos, rotation=45)
    plt.xlabel('Tipo de dato')
    plt.ylabel('Memoria promedio (MB)')
    plt.title(f'Memoria promedio por tipo de dato CPU vs CPU - Matriz {tamano}x{tamano}\n i5_10400_4.10GHz vs i7_10810u')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Guardar y mostrar gráfico
    plt.savefig(f"output/vsprocesadores/memoriaCPU/memoria_comparativo_{tamano}.jpg")
    plt.show()
    plt.close()

    

# --- FUNCIÓN PARA GRAFICAR GPU vs GPU (CUDA vs OpenCL) ---
def graficar_tiempo_comparativo_GPUvsGPU(tamano):
    # Leer archivos base
    df_cudaIvan = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/CUDA/cuda_results_IVAN.csv")
    df_openclIvan = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/OpenCL/opencl_resultsIvan.csv")
    df_cudaAlex = pd.read_csv("csv/intel core i7-10810u/CUDA/cuda_resultsnew.csv")
    df_openclAlex = pd.read_csv("csv/intel core i7-10810u/OpenCL/opencl_results.csv")
    
    # Generar DataFrames procesados
    df_cudaIvan_res = generar_df_tiempo(df_cudaIvan, size=tamano)
    df_openclIvan_res = generar_df_tiempo(df_openclIvan, size=tamano)
    df_cudaAlex_res = generar_df_tiempo(df_cudaAlex, size=tamano)
    df_openclAlex_res = generar_df_tiempo(df_openclAlex, size=tamano)
    
    # Calcular promedios
    promedios_cudaIvan = df_cudaIvan_res.iloc[:, 1:].mean(axis=1)
    promedios_openclIvan = df_openclIvan_res.iloc[:, 1:].mean(axis=1)
    promedios_cudaAlex = df_cudaAlex_res.iloc[:, 1:].mean(axis=1)
    promedios_openclAlex = df_openclAlex_res.iloc[:, 1:].mean(axis=1)
    tipos = df_cudaIvan_res["DataType"]

    # Configuración de barras
    plt.figure(figsize=(12, 7))
    x = np.arange(len(tipos))
    width = 0.2  # Ancho de cada grupo de barras

    # Graficar cada grupo de barras
    plt.bar(x - 1.5*width, promedios_cudaIvan, width=width, label='CUDA - i5_10400_4.10GHz_12hilos', color='gold')
    plt.bar(x - 0.5*width, promedios_openclIvan, width=width, label='OpenCL - i5_10400_4.10GHz_12hilos', color='orange')
    plt.bar(x + 0.5*width, promedios_cudaAlex, width=width, label='CUDA - i7-10810u', color='limegreen')
    plt.bar(x + 1.5*width, promedios_openclAlex, width=width, label='OpenCL - i7-10810u', color='green')

    # Personalizar ejes
    plt.xticks(x, tipos, rotation=45)
    plt.xlabel('Tipo de dato')
    plt.ylabel('Tiempo promedio (s)')
    plt.title(f'Tiempo promedio por tipo de dato - Matriz {tamano}x{tamano}\n GPU (CUDA vs OpenCL)')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Guardar y mostrar gráfico
    plt.savefig(f"output/vsprocesadores/tiempoGPU/tiempo_comparativo_{tamano}.jpg")
    plt.show()
    plt.close()


# --- FUNCIÓN PARA GRAFICAR GPU vs GPU (CUDA vs OpenCL) ---
def graficar_memoria_comparativo_GPUvsGPU(tamano):
    # Leer archivos base
    df_cudaIvan = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/CUDA/cuda_results_IVAN.csv")
    df_openclIvan = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/OpenCL/opencl_resultsIvan.csv")
    df_cudaAlex = pd.read_csv("csv/intel core i7-10810u/CUDA/cuda_resultsnew.csv")
    df_openclAlex = pd.read_csv("csv/intel core i7-10810u/OpenCL/opencl_results.csv")
    
    # Generar DataFrames procesados
    df_cudaIvan_res = generar_df_memoria(df_cudaIvan, size=tamano)
    df_openclIvan_res = generar_df_memoria(df_openclIvan, size=tamano)
    df_cudaAlex_res = generar_df_memoria(df_cudaAlex, size=tamano)
    df_openclAlex_res = generar_df_memoria(df_openclAlex, size=tamano)
    
    # Calcular promedios
    promedios_cudaIvan = df_cudaIvan_res.iloc[:, 1:].mean(axis=1)
    promedios_openclIvan = df_openclIvan_res.iloc[:, 1:].mean(axis=1)
    promedios_cudaAlex = df_cudaAlex_res.iloc[:, 1:].mean(axis=1)
    promedios_openclAlex = df_openclAlex_res.iloc[:, 1:].mean(axis=1)
    tipos = df_cudaIvan_res["DataType"]

    # Configuración de barras
    plt.figure(figsize=(12, 7))
    x = np.arange(len(tipos))
    width = 0.2  # Ancho de cada grupo de barras

    # Graficar cada grupo de barras
    plt.bar(x - 1.5*width, promedios_cudaIvan, width=width, label='CUDA - i5_10400_4.10GHz_12hilos', color='gold')
    plt.bar(x - 0.5*width, promedios_openclIvan, width=width, label='OpenCL - i5_10400_4.10GHz_12hilos', color='orange')
    plt.bar(x + 0.5*width, promedios_cudaAlex, width=width, label='CUDA - i7-10810u', color='limegreen')
    plt.bar(x + 1.5*width, promedios_openclAlex, width=width, label='OpenCL - i7-10810u', color='green')

    # Personalizar ejes
    plt.xticks(x, tipos, rotation=45)
    plt.xlabel('Tipo de dato')
    plt.ylabel('Memoria promedio (MB)')
    plt.title(f'Memoria promedio por tipo de dato - Matriz {tamano}x{tamano}\n GPU (CUDA vs OpenCL)')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Guardar y mostrar gráfico
    plt.savefig(f"output/vsprocesadores/memoriaGPU/tiempo_comparativo_{tamano}.jpg")
    plt.show()
    plt.close()


# --- FUNCIÓN PARA GRAFICAR CUDA con ThreadPercentage ---
def graficar_tiempo_CUDA_thread_percentage(tamano):
    # Leer archivos base
    df_cudaIvan = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/CUDA/cuda_results_IVAN.csv")
    df_cudaAlex = pd.read_csv("csv/intel core i7-10810u/CUDA/cuda_resultsnew.csv")
    
    # Generar DataFrames procesados
    df_cudaIvan_res = generar_df_tiempo_cuda(df_cudaIvan, size=tamano)
    df_cudaAlex_res = generar_df_tiempo_cuda(df_cudaAlex, size=tamano)
    
    # Filtrar porcentajes únicos
    porcentajes = sorted(df_cudaIvan_res["ThreadPercentage"].unique())
    
    # Graficar
    plt.figure(figsize=(15, 8))
    width = 0.35  # Ancho de las barras
    x = np.arange(len(porcentajes))  # Posiciones en X

    # Calcular promedios para cada porcentaje
    promedios_ivan = [df_cudaIvan_res[df_cudaIvan_res["ThreadPercentage"] == p].iloc[:, 2:].mean(axis=1).mean() for p in porcentajes]
    promedios_alex = [df_cudaAlex_res[df_cudaAlex_res["ThreadPercentage"] == p].iloc[:, 2:].mean(axis=1).mean() for p in porcentajes]
    
    # Graficar barras
    plt.bar(x - width/2, promedios_ivan, width=width, color='gold', label='CUDA - i5_10400_4.10GHz_12hilos')
    plt.bar(x + width/2, promedios_alex, width=width, color='limegreen', label='CUDA - i7-10810u')

    # Etiquetas y formato
    plt.xticks(x, [f"{p}%" for p in porcentajes], rotation=45)
    plt.xlabel('Porcentaje de hilos (ThreadPercentage)')
    plt.ylabel('Tiempo promedio (s)')
    plt.title(f'Tiempo promedio CUDA por ThreadPercentage - Matriz {tamano}x{tamano}')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Guardar y mostrar gráfico
    plt.savefig(f"output/PorcentajesCuda/tiempo/cuda_thread_percentage_{tamano}.jpg")
    plt.show()
    plt.close()

def graficar_memoria_CUDA_thread_percentage(tamano):
    # Leer archivos base
    df_cudaIvan = pd.read_csv("csv/i5_10400_4.10GHz_12hilos/CUDA/cuda_results_IVAN.csv")
    df_cudaAlex = pd.read_csv("csv/intel core i7-10810u/CUDA/cuda_resultsnew.csv")
    
    # Generar DataFrames procesados
    df_cudaIvan_res = generar_df_memoria_cuda(df_cudaIvan, size=tamano)
    df_cudaAlex_res = generar_df_memoria_cuda(df_cudaAlex, size=tamano)
    
    # Filtrar porcentajes únicos
    porcentajes = sorted(df_cudaIvan_res["ThreadPercentage"].unique())
    
    # Graficar
    plt.figure(figsize=(15, 8))
    width = 0.35  # Ancho de las barras
    x = np.arange(len(porcentajes))  # Posiciones en X

    # Calcular promedios para cada porcentaje
    promedios_ivan = [df_cudaIvan_res[df_cudaIvan_res["ThreadPercentage"] == p].iloc[:, 2:].mean(axis=1).mean() for p in porcentajes]
    promedios_alex = [df_cudaAlex_res[df_cudaAlex_res["ThreadPercentage"] == p].iloc[:, 2:].mean(axis=1).mean() for p in porcentajes]
    
    # Graficar barras
    plt.bar(x - width/2, promedios_ivan, width=width, color='gold', label='CUDA - i5_10400_4.10GHz_12hilos')
    plt.bar(x + width/2, promedios_alex, width=width, color='limegreen', label='CUDA - i7-10810u')

    # Etiquetas y formato
    plt.xticks(x, [f"{p}%" for p in porcentajes], rotation=45)
    plt.xlabel('Porcentaje de hilos (ThreadPercentage)')
    plt.ylabel('Memoria promedio (MB)')
    plt.title(f'Memoria promedio CUDA por ThreadPercentage - Matriz {tamano}x{tamano}')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Guardar y mostrar gráfico
    plt.savefig(f"output/PorcentajesCuda/memoria/cuda_thread_percentage_{tamano}.jpg")
    plt.show()
    plt.close()

# --- para ejecutar solamente quitar el simbolo que provee el comentario ---
sizes = [64,128,256,512,1024,2048,4096,8192,16384]
for size in sizes:
    #graficar_tiempo_comparativo(size)
    #graficar_tiempo_sec_openmp(size)
    #graficar_memoria_comparativo(size)
    #graficar_memoria_sec_openmp(size)
    #graficar_por_tipos_dato_por_grupo_tiempo()
    #graficar_por_tipos_dato_por_grupo_memoria()
    #graficar_cpu_vs_gpu_por_tamaño_memoria()
    #graficar_cpu_vs_gpu_por_tamaño_tiempo()
    #graficar_comparativa("gpu", rango_tamanos)
    #graficar_comparativa("cpu", rango_tamanos)
    #graficar_tiempo_comparativo_CPUvsCPU(size)
    #graficar_memoria_comparativo_CPUvsCPU(size)
    #graficar_tiempo_comparativo_GPUvsGPU(size)
    #graficar_memoria_comparativo_GPUvsGPU(size)
    graficar_tiempo_CUDA_thread_percentage(size)
    graficar_memoria_CUDA_thread_percentage(size)