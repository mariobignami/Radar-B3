import yfinance as yf
from datetime import datetime

try:
    mglu = yf.Ticker('MGLU3.SA')
    hist = mglu.history(start='2026-04-30', end='2026-05-01')
    
    if not hist.empty:
        today = hist.iloc[-1]
        print('MGLU3 - VALORES REAIS DE 30/04/2026:')
        print(f'  Preco Abertura: R$ {today["Open"]:.2f}')
        print(f'  Preco Maximo: R$ {today["High"]:.2f}')
        print(f'  Preco Minimo: R$ {today["Low"]:.2f}')
        print(f'  Preco Fechamento: R$ {today["Close"]:.2f}')
        print(f'  Volume: {today["Volume"] / 1_000_000:.2f} milhoes')
        
        # Dados da empresa
        print(f'\n  Setor: Comercio')
        print(f'  Segmento: Varejo')
        print(f'  Mes: 4 (Abril)')
        
        # Qual era o dia da semana em 2026-04-30?
        date_obj = datetime.strptime('2026-04-30', '%Y-%m-%d')
        day_of_week = date_obj.weekday()  # 0=Monday, 4=Friday
        day_names = ['Segunda-feira', 'Terca-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sabado', 'Domingo']
        print(f'  Dia da Semana: {day_of_week} ({day_names[day_of_week]})')
    else:
        print('Sem dados disponíveis')
except Exception as e:
    print(f'Erro: {e}')
