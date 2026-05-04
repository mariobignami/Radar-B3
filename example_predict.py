"""
Script de exemplo para fazer predições usando o modelo treinado
Demonstra como usar a classe StockPredictor
"""
from src.predict import StockPredictor
from src.config import Config

def main():
    print("=" * 60)
    print("🔮 PREVISÃO DE PREÇOS DE AÇÕES B3")
    print("=" * 60)
    
    # 1️⃣ Carregar o modelo
    print("\n📊 Carregando modelo...")
    predictor = StockPredictor('linear_regression_model.pkl')
    
    # 2️⃣ Fazer predições individuais
    print("\n" + "=" * 60)
    print("📈 Exemplos de Predição")
    print("=" * 60)
    
    # Exemplo 1: Petrobras
    print("\n1️⃣  PETR4 (Petrobras)")
    result1 = predictor.predict_single(
        open_price=28.5,
        high_price=29.2,
        low_price=28.1,
        quantity=500_000_000,
        sector='Energia',
        segment='Petróleo',
        month=4,
        day_week=1
    )
    print(f"   Entrada: Abertura R$ {result1['input']['open']}")
    print(f"   Predição: R$ {result1['predicted_price']}")
    
    # Exemplo 2: Vale
    print("\n2️⃣  VALE3 (Vale S.A.)")
    result2 = predictor.predict_single(
        open_price=59.8,
        high_price=61.2,
        low_price=59.5,
        quantity=300_000_000,
        sector='Mineração',
        segment='Ferro',
        month=4,
        day_week=2
    )
    print(f"   Entrada: Abertura R$ {result2['input']['open']}")
    print(f"   Predição: R$ {result2['predicted_price']}")
    
    # Exemplo 3: Itaú
    print("\n3️⃣  ITUB4 (Itaú Unibanco)")
    result3 = predictor.predict_single(
        open_price=11.5,
        high_price=11.9,
        low_price=11.4,
        quantity=400_000_000,
        sector='Financeiro',
        segment='Banco',
        month=4,
        day_week=3
    )
    print(f"   Entrada: Abertura R$ {result3['input']['open']}")
    print(f"   Predição: R$ {result3['predicted_price']}")
    
    # 3️⃣ Fazer predição customizada
    print("\n" + "=" * 60)
    print("🎯 Predição Customizada")
    print("=" * 60)
    
    print("\nDigite os parâmetros para uma predição:")
    try:
        open_p = float(input("   Preço de abertura (ex: 30.5): ") or "30.5")
        high_p = float(input("   Preço máximo (ex: 31.2): ") or "31.2")
        low_p = float(input("   Preço mínimo (ex: 30.1): ") or "30.1")
        volume = float(input("   Volume/Quantidade (ex: 500000000): ") or "500000000")
        
        result = predictor.predict_single(
            open_price=open_p,
            high_price=high_p,
            low_price=low_p,
            quantity=volume,
            sector='Energia'
        )
        
        print(f"\n   ✅ Predição: R$ {result['predicted_price']}")
    except ValueError:
        print("   ❌ Entrada inválida. Usando valores padrão.")
    
    # Resumo
    print("\n" + "=" * 60)
    print("✅ Predições concluídas!")
    print("=" * 60)
    print("\n💡 Dicas:")
    print("   • Use predict_single() para predições individuais")
    print("   • Use predict_batch(df) para predições em lote")
    print("   • O modelo espera dados no formato correto (veja Config)")
    print(f"   • Modelo salvo em: {Config.MODEL_DIR}")

if __name__ == "__main__":
    main()
