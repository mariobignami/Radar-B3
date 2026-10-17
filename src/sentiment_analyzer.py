"""
Módulo de Análise de Sentimento de Mercado
Coleta dados de Google Trends e RSS feeds para análise de sentimento
"""
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup


class SentimentAnalyzer:
    """Analisa sentimento de mercado usando fontes gratuitas."""
    
    def __init__(self):
        self.pytrends = TrendReq(hl='pt-BR', tz=360)
    
    def get_google_trends(self, keywords: List[str], timeframe: str = 'today 3-m') -> pd.DataFrame:
        """
        Coleta dados do Google Trends para palavras-chave.
        
        Args:
            keywords: Lista de palavras-chave (ex: ['PETR4', 'VALE3', 'Ibovespa'])
            timeframe: Período de tempo (default: últimos 3 meses)
        
        Returns:
            DataFrame com dados de tendência
        """
        try:
            self.pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='BR', gprop='')
            data = self.pytrends.interest_over_time()
            
            if data.empty:
                print(f"⚠️ Nenhum dado do Google Trends para {keywords}")
                return pd.DataFrame()
            
            return data.drop(columns=['isPartial'], errors='ignore')
        except Exception as e:
            print(f"❌ Erro ao buscar Google Trends: {e}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, rss_urls: List[str], ticker: Optional[str] = None) -> List[Dict]:
        """
        Coleta notícias de RSS feeds e analisa sentimento básico.
        
        Args:
            rss_urls: Lista de URLs de RSS feeds
            ticker: Ticker específico para filtrar (opcional)
        
        Returns:
            Lista de dicionários com notícias e metadados
        """
        news_items = []
        
        for url in rss_urls:
            try:
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:20]:  # Últimas 20 notícias
                    title = entry.get('title', '')
                    summary = entry.get('summary', entry.get('description', ''))
                    published = entry.get('published', '')
                    
                    # Filtrar por ticker se especificado
                    if ticker and ticker.lower() not in title.lower():
                        continue
                    
                    # Análise de sentimento simples baseada em palavras-chave
                    sentiment = self._analyze_text_sentiment(title + ' ' + summary)
                    
                    news_items.append({
                        'title': title,
                        'summary': summary[:200],  # Primeiros 200 caracteres
                        'published': published,
                        'sentiment': sentiment,
                        'source': url
                    })
            except Exception as e:
                print(f"⚠️ Erro ao ler RSS {url}: {e}")
        
        return news_items
    
    def _analyze_text_sentiment(self, text: str) -> str:
        """
        Análise de sentimento simples baseada em palavras-chave.
        Em produção, usar biblioteca como NLTK ou spaCy.
        """
        positive_words = [
            'alta', 'ganho', 'lucro', 'crescimento', 'recorde', 'aumento',
            'forte', 'positivo', 'expansão', 'investimento', 'boom',
            'recuperação', 'subiu', 'valorizou', 'superou'
        ]
        
        negative_words = [
            'queda', 'perda', 'prejuízo', 'crise', 'redução', 'diminuição',
            'fraco', 'negativo', 'recessão', 'venda', 'queda livre',
            'desvalorizou', 'perdeu', 'abaixou', 'preocupação'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def calculate_sentiment_score(self, news_items: List[Dict]) -> Dict:
        """
        Calcula score de sentimento agregado de notícias.
        
        Args:
            news_items: Lista de notícias com sentimento
        
        Returns:
            Dicionário com métricas de sentimento
        """
        if not news_items:
            return {
                'total_news': 0,
                'positive_pct': 0,
                'negative_pct': 0,
                'neutral_pct': 0,
                'overall_sentiment': 'neutral'
            }
        
        total = len(news_items)
        positive = sum(1 for item in news_items if item['sentiment'] == 'positive')
        negative = sum(1 for item in news_items if item['sentiment'] == 'negative')
        neutral = total - positive - negative
        
        # Determinar sentimento geral
        if positive > negative * 1.5:
            overall = 'strongly_positive'
        elif positive > negative:
            overall = 'positive'
        elif negative > positive * 1.5:
            overall = 'strongly_negative'
        elif negative > positive:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        return {
            'total_news': total,
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral,
            'positive_pct': (positive / total) * 100,
            'negative_pct': (negative / total) * 100,
            'neutral_pct': (neutral / total) * 100,
            'overall_sentiment': overall
        }
    
    def get_reddit_sentiment(self, subreddit: str = 'investimentos', limit: int = 50) -> List[Dict]:
        """
        Coleta posts do Reddit para análise de sentimento.
        Nota: Requer API key do Reddit para uso produtivo.
        Esta é uma implementação simplificada.
        
        Args:
            subreddit: Nome do subreddit
            limit: Número de posts a coletar
        
        Returns:
            Lista de posts com sentimento
        """
        # Implementação placeholder - em produção usar PRAW com API key
        print("⚠️ Reddit scraping requer API key. Implementação placeholder.")
        return []
    
    def get_comprehensive_sentiment(self, ticker: str) -> Dict:
        """
        Retorna análise de sentimento completa para um ticker.
        
        Args:
            ticker: Ticker da ação (ex: 'PETR4')
        
        Returns:
            Dicionário com todas as métricas de sentimento
        """
        result = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'google_trends': {},
            'news_sentiment': {},
            'overall_score': 0
        }
        
        # Google Trends
        trends_data = self.get_google_trends([ticker, f'{ticker}.SA'])
        if not trends_data.empty:
            latest_trend = trends_data.iloc[-1]
            result['google_trends'] = {
                'trend_score': latest_trend.get(ticker, 0),
                'trend_7d_avg': trends_data[ticker].tail(7).mean() if ticker in trends_data.columns else 0
            }
        
        # Notícias
        rss_urls = [
            'https://www.infomoney.com.br/rss/mercados/',
            'https://www.valor.globo.com/rss/valor-investimentos/',
            'https://economia.uol.com.br/ultimas/index.xml'
        ]
        
        news_items = self.get_news_sentiment(rss_urls, ticker)
        news_sentiment = self.calculate_sentiment_score(news_items)
        result['news_sentiment'] = news_sentiment
        
        # Score geral (simplificado)
        trend_score = result['google_trends'].get('trend_score', 0) / 100
        news_score = (news_sentiment['positive_pct'] - news_sentiment['negative_pct']) / 100
        result['overall_score'] = (trend_score + news_score) / 2
        
        return result


# RSS feeds brasileiros com notícias financeiras
BRAZILIAN_FINANCE_RSS = [
    'https://www.infomoney.com.br/rss/mercados/',
    'https://www.valor.globo.com/rss/valor-investimentos/',
    'https://economia.uol.com.br/ultimas/index.xml',
    'https://www1.folha.uol.com.br/mercado/rss09.xml'
]


def get_sentiment_for_tickers(tickers: List[str]) -> pd.DataFrame:
    """
    Função helper para coletar sentimento para múltiplos tickers.
    
    Args:
        tickers: Lista de tickers (ex: ['PETR4', 'VALE3', 'ITUB4'])
    
    Returns:
        DataFrame com dados de sentimento
    """
    analyzer = SentimentAnalyzer()
    results = []
    
    for ticker in tickers:
        try:
            sentiment = analyzer.get_comprehensive_sentiment(ticker)
            results.append(sentiment)
        except Exception as e:
            print(f"❌ Erro ao analisar sentimento para {ticker}: {e}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Teste com PETR4
    analyzer = SentimentAnalyzer()
    
    print("=== Teste Google Trends ===")
    trends = analyzer.get_google_trends(['PETR4', 'VALE3'])
    if not trends.empty:
        print(f"Shape: {trends.shape}")
        print(f"Últimos dados:\n{trends.tail()}")
    
    print("\n=== Teste Notícias ===")
    news = analyzer.get_news_sentiment(BRAZILIAN_FINANCE_RSS[:2], 'PETR4')
    print(f"Notícias encontradas: {len(news)}")
    if news:
        print(f"Primeira notícia: {news[0]['title']}")
    
    print("\n=== Teste Sentimento Completo ===")
    sentiment = analyzer.get_comprehensive_sentiment('PETR4')
    print(f"Sentimento para PETR4:")
    print(f"  Google Trends: {sentiment['google_trends']}")
    print(f"  Notícias: {sentiment['news_sentiment']}")
    print(f"  Score Geral: {sentiment['overall_score']:.2f}")
