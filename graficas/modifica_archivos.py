input_file = "Datos_final_sin_corregir.txt"
output_file = "Datos_finales.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = []
    for line in f:
        # Eliminar comillas, quitar espacios extra, reemplazar tabs por comas
        clean_line = line.strip().replace('"', '').replace('\t', ',')
        lines.append(clean_line)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Archivo convertido guardado como: {output_file}")
