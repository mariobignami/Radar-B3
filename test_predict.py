"""
Script não-interativo para testar as predições
"""
from src.predict import StockPredictor

def main():
    print("=" * 60)
    print("🔮 PREVISÃO DE PREÇOS DE AÇÕES B3")
    print("=" * 60)
    
    # Carregar o modelo
    print("\n📊 Carregando modelo...")
    predictor = StockPredictor('linear_regression_model.pkl')
    
    # Fazer predições
    print("\n" + "=" * 60)
    print("📈 Exemplos de Predição")
    print("=" * 60)
    
    examples = [
        {
            'name': 'PETR4 (Petrobras)',
            'code': 'PETR4',
            'open': 28.5, 'high': 29.2, 'low': 28.1, 'qty': 500_000_000,
            'sector': 'Energia', 'segment': 'Petróleo'
        },
        {
            'name': 'VALE3 (Vale)',
            'code': 'VALE3',
            'open': 59.8, 'high': 61.2, 'low': 59.5, 'qty': 300_000_000,
            'sector': 'Mineração', 'segment': 'Ferro'
        },
        {
            'name': 'ITUB4 (Itaú)',
            'code': 'ITUB4',
            'open': 11.5, 'high': 11.9, 'low': 11.4, 'qty': 400_000_000,
            'sector': 'Financeiro', 'segment': 'Banco'
        },
        {
            'name': 'ABEV3 (Ambev)',
            'code': 'ABEV3',
            'open': 10.2, 'high': 10.8, 'low': 10.1, 'qty': 200_000_000,
            'sector': 'Bebidas', 'segment': 'Bebidas'
        },
        {
            'name': 'MGLU3 (Magazine Luiza)',
            'code': 'MGLU3',
            'open': 8.5, 'high': 9.2, 'low': 8.3, 'qty': 150_000_000,
            'sector': 'Comércio', 'segment': 'Varejo'
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\n{i}️⃣  {ex['name']}")
        result = predictor.predict_single(
            open_price=ex['open'],
            high_price=ex['high'],
            low_price=ex['low'],
            quantity=ex['qty'],
            stock_code=ex['code'],
            sector=ex['sector'],
            segment=ex['segment']
        )
        print(f"   📊 Abertura: R$ {ex['open']:.2f}")
        print(f"   🎯 Fechamento Previsto: R$ {result['predicted_price']:.2f}")
        if result['predicted_price'] > ex['open']:
            variacao = ((result['predicted_price'] - ex['open']) / ex['open'] * 100)
            print(f"   📈 Variação: +{variacao:.2f}%")
        else:
            variacao = ((result['predicted_price'] - ex['open']) / ex['open'] * 100)
            print(f"   📉 Variação: {variacao:.2f}%")
    
    print("\n" + "=" * 60)
    print("✅ Predições concluídas com sucesso!")
    print("=" * 60)

if __name__ == "__main__":
    main()
