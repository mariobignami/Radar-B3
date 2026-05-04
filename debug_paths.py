from pathlib import Path

base = Path(__file__).resolve().parent
raw_data = base / "data" / "raw"

print(f"Diretório Raiz: {base}")
print(f"Pasta de Dados: {raw_data}")
print(f"Arquivos encontrados: {[f.name for f in raw_data.glob('*.csv')]}")