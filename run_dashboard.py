#!/usr/bin/env python
"""
Script para executar o dashboard Streamlit
"""
import subprocess
import sys

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Iniciando Dashboard B3 Stock Prediction")
    print("=" * 70)
    print("\n📍 Abra seu navegador em: http://localhost:8501")
    print("\n(Pressione Ctrl+C para parar)\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard finalizado")
        sys.exit(0)
