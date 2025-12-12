"""
V20 Constituent Analyzer - Main Script
Analyse quotidienne des composants d'indices (copie de V19 adaptée)
"""

import sys
sys.path.append('gemini_v15')  # Pour indicateurs

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import schedule
import time
from datetime import datetime

from gemini_v20_invest.data.stock_loader import StockDataLoader
from gemini_v20_invest.utils.telegram_notifier import TelegramNotifier
from gemini_v20_invest.data.v20_database import V20Database
from gemini_v15.utils.indicators import Indicators
from gemini_v20_invest.live.universe_v20 import INDICES_UNIVERSE, ANALYSIS_TIME

class V20ConstituentAnalyzer:
    """
    Analyseur de composants d'indices (similaire à AlphaZeroTrader de V19)
    """
    def __init__(self, symbol):
        self.symbol = symbol
        self.loader = StockDataLoader()
    
    def analyze(self):
        """
        Analyse une action et retourne un signal
        (Copie simplifiée de la logique V19)
        """
        # 1. Fetch data D1
        df = self.loader.get_data(self.symbol, period='1y', interval='1d')
        if df is None:
            return None
        
        # Normalize columns
        df.columns = df.columns.str.lower()
        if 'volume' in df.columns:
            df['tick_volume'] = df['volume']
        df['spread'] = 0
        
        # 2. Apply 26 indicators (comme V19)
        df = Indicators.add_all(df)
        
        # 3. Get latest values
        latest = df.iloc[-1]
        last_price = latest['close']
        
        # Key indicators
        rsi = latest.get('rsi', 50)
        stoch = latest.get('stoch_k', 0.5)
        bb_b = latest.get('bb_b', 0.5)
        macd_signal = latest.get('macd_signal', 0)
        adx = latest.get('adx', 0)
        
        # 4. Simple decision logic (comme POC V20)
        # TODO: Remplacer par MCTS une fois le modèle entraîné
        signal = None
        
        if rsi < 35 and stoch < 0.3:
            signal = 'BULLISH'
            confidence = 0.85
        elif rsi < 45 and bb_b < 0.3:
            signal = 'BULLISH'
            confidence = 0.65
        elif rsi > 65 and stoch > 0.7:
            signal = 'BEARISH'
            confidence = 0.85
        elif rsi > 55 and bb_b > 0.7:
            signal = 'BEARISH'
            confidence = 0.65
        else:
            signal = 'NEUTRAL'
            confidence = 0.50
        
        return {
            'symbol': self.symbol,
            'signal': signal,
            'confidence': confidence,
            'price': last_price,
            'rsi': rsi,
            'stoch': stoch * 100,
            'adx': adx
        }

class V20Orchestrator:
    """
    Orchestrateur V20 (similaire à MultiSymbolOrchestrator de V19)
    """
    def __init__(self):
        self.telegram = TelegramNotifier()
        self.db = V20Database()
        print("[OK] V20 Orchestrator initialized")
    
    def analyze_index(self, index_code, config):
        """
        Analyse tous les composants d'un indice
        """
        print(f"\n{'='*60}")
        print(f"📊 Analyzing {config['name']} ({index_code})")
        print(f"{'='*60}")
        
        symbols = config['symbols']
        results = {
            'bullish': [],
            'bearish': [],
            'neutral': []
        }
        
        # Analyse chaque composant
        for symbol in symbols:
            try:
                analyzer = V20ConstituentAnalyzer(symbol)
                analysis = analyzer.analyze()
                
                if analysis:
                    if analysis['signal'] == 'BULLISH':
                        results['bullish'].append(analysis)
                    elif analysis['signal'] == 'BEARISH':
                        results['bearish'].append(analysis)
                    else:
                        results['neutral'].append(analysis)
                    
                    print(f"  {symbol}: {analysis['signal']} ({analysis['confidence']*100:.0f}%)")
            except Exception as e:
                print(f"  {symbol}: ERROR - {e}")
        
        # Aggregate results
        total = len(symbols)
        bullish_count = len(results['bullish'])
        bearish_count = len(results['bearish'])
        neutral_count = len(results['neutral'])
        
        # Determine index direction
        bullish_pct = bullish_count / total if total > 0 else 0
        bearish_pct = bearish_count / total if total > 0 else 0
        
        if bullish_pct > 0.6:
            index_signal = 'HAUSSIER ↗️'
            confidence = bullish_pct
        elif bearish_pct > 0.6:
            index_signal = 'BAISSIER ↘️'
            confidence = bearish_pct
        else:
            index_signal = 'NEUTRE ↔️'
            confidence = 0.5
        
        # Summary
        summary = {
            'index': index_code,
            'name': config['name'],
            'signal': index_signal,
            'confidence': confidence,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total': total,
            'top_bullish': sorted(results['bullish'], key=lambda x: x['confidence'], reverse=True)[:5],
            'top_bearish': sorted(results['bearish'], key=lambda x: x['confidence'], reverse=True)[:5]
        }
        
        print(f"\n✅ {config['name']}: {index_signal} (Confiance: {confidence*100:.0f}%)")
        print(f"   Bullish: {bullish_count}/{total} ({bullish_pct*100:.0f}%)")
        print(f"   Bearish: {bearish_count}/{total} ({bearish_pct*100:.0f}%)")
        
        return summary
    
    def send_daily_report(self, analyses):
        """
        Envoyer le rapport quotidien via Telegram
        """
        message = f"""
📊 <b>V20 CONSTITUENT ANALYSIS</b>
<b>Date:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}

<b>═══ PRÉDICTIONS INDICES ═══</b>
"""
        
        for analysis in analyses:
            message += f"""
<b>{analysis['name']} ({analysis['index']})</b>
Direction: {analysis['signal']}
Confiance: {analysis['confidence']*100:.0f}%
Composants: {analysis['bullish_count']} Bull / {analysis['bearish_count']} Bear / {analysis['neutral_count']} Neutre
"""
            
            # Top movers
            if analysis['top_bullish']:
                message += "\n📈 <b>Top Bullish:</b>\n"
                for stock in analysis['top_bullish'][:3]:
                    message += f"  • {stock['symbol']}: RSI {stock['rsi']:.0f}\n"
            
            if analysis['top_bearish']:
                message += "\n📉 <b>Top Bearish:</b>\n"
                for stock in analysis['top_bearish'][:3]:
                    message += f"  • {stock['symbol']}: RSI {stock['rsi']:.0f}\n"
        
        message += f"\n<i>Analyse générée par V20 AlphaZero</i>"
        
        self.telegram.send_report("V20 Daily Analysis", message)
    
    def run_daily_analysis(self):
        """
        Lancer l'analyse quotidienne (appelé par scheduler)
        """
        print(f"\n🕐 Starting V20 Daily Analysis - {datetime.now()}")
        
        analyses = []
        for index_code, config in INDICES_UNIVERSE.items():
            analysis = self.analyze_index(index_code, config)
            analyses.append(analysis)
        
        # Send Telegram report
        self.send_daily_report(analyses)
        
        print(f"\n✅ Analysis complete - Report sent via Telegram")
    
    def start_scheduler(self):
        """
        Démarrer le scheduler (analyse à 20h tous les jours)
        """
        schedule.every().day.at(ANALYSIS_TIME).do(self.run_daily_analysis)
        
        print(f"🕐 V20 Scheduler started - Daily analysis at {ANALYSIS_TIME}")
        print(f"📊 Monitoring {len(INDICES_UNIVERSE)} indices")
        
        # Run once immediately for testing
        print("\n🧪 Running initial analysis for testing...")
        self.run_daily_analysis()
        
        # Then wait for scheduled time
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    orchestrator = V20Orchestrator()
    orchestrator.start_scheduler()
