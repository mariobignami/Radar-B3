"""
Script para fazer predições em datas específicas
Uso: python predict_date.py 2026-04-30
"""
import sys
from datetime import datetime
from src.predict import StockPredictor

def predict_for_date(date_str):
    """
    Faz predições para uma data específica.
    
    Args:
        date_str (str): Data no formato YYYY-MM-DD
    """
    try:
        # Parser da data
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        month = date_obj.month
        day = date_obj.day
        day_week = date_obj.weekday()  # 0=seg, 1=ter, 2=qua, 3=qui, 4=sex, 5=sab, 6=dom
        
        # Nomes dos dias
        days_name = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        months_name = ['', 'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                       'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
        
        print("=" * 70)
        print(f"🔮 PREVISÃO DE PREÇOS - {day:02d} de {months_name[month]} de {date_obj.year}")
        print(f"   ({days_name[day_week]})")
        print("=" * 70)
        
        # Carregar preditor
        predictor = StockPredictor('linear_regression_model.pkl')
        
        # Dados de cada ação
        stocks = [
            {
                'name': 'PETR4',
                'company': 'Petrobras',
                'open': 28.5, 'high': 29.2, 'low': 28.1, 'qty': 500_000_000,
                'sector': 'Energia', 'segment': 'Petróleo'
            },
            {
                'name': 'VALE3',
                'company': 'Vale',
                'open': 59.8, 'high': 61.2, 'low': 59.5, 'qty': 300_000_000,
                'sector': 'Mineração', 'segment': 'Ferro'
            },
            {
                'name': 'ITUB4',
                'company': 'Itaú',
                'open': 11.5, 'high': 11.9, 'low': 11.4, 'qty': 400_000_000,
                'sector': 'Financeiro', 'segment': 'Banco'
            },
            {
                'name': 'ABEV3',
                'company': 'Ambev',
                'open': 10.2, 'high': 10.8, 'low': 10.1, 'qty': 200_000_000,
                'sector': 'Bebidas', 'segment': 'Bebidas'
            },
            {
                'name': 'MGLU3',
                'company': 'Magazine Luiza',
                'open': 8.5, 'high': 9.2, 'low': 8.3, 'qty': 150_000_000,
                'sector': 'Comércio', 'segment': 'Varejo'
            }
        ]
        
        print("\n")
        for i, stock in enumerate(stocks, 1):
            result = predictor.predict_single(
                open_price=stock['open'],
                high_price=stock['high'],
                low_price=stock['low'],
                quantity=stock['qty'],
                sector=stock['sector'],
                segment=stock['segment'],
                month=month,
                day_week=day_week
            )
            
            if result['predicted_price']:
                variacao = ((result['predicted_price'] - stock['open']) / stock['open'] * 100)
                sinal = "📈" if variacao > 0 else "📉"
                
                print(f"{i}️⃣  {stock['name']} ({stock['company']})")
                print(f"    Abertura:    R$ {stock['open']:.2f}")
                print(f"    Previsão:    R$ {result['predicted_price']:.2f}")
                print(f"    {sinal} Variação: {variacao:+.2f}%")
                print()
        
        print("=" * 70)
        print("✅ Predições para esta data concluídas!")
        print("=" * 70)
        
    except ValueError:
        print(f"❌ Erro: Data inválida '{date_str}'")
        print("💡 Use o formato: YYYY-MM-DD (ex: 2026-04-30)")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict_date.py YYYY-MM-DD")
        print("Exemplo: python predict_date.py 2026-04-30")
        sys.exit(1)
    
    date_input = sys.argv[1]
    predict_for_date(date_input)
