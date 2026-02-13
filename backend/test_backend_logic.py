import unittest
import sys
from unittest.mock import MagicMock, patch

# Mock MetaTrader5 before importing gemini_core
sys.modules['MetaTrader5'] = MagicMock()

from datetime import datetime, timedelta
from gemini_core import GeminiCore
from database import Trade

class TestGeminiCore(unittest.TestCase):
    def setUp(self):
        self.gemini = GeminiCore()
        # Mock the database session and query
        self.mock_db = MagicMock()
        self.mock_query = MagicMock()
        self.mock_db.query.return_value = self.mock_query
        self.mock_query.filter.return_value = self.mock_query # Chain filters

    @patch('gemini_core.SessionLocal')
    def test_get_history_analysis_empty(self, mock_session_local):
        mock_session_local.return_value = self.mock_db
        self.mock_query.all.return_value = []

        result = self.gemini.get_history_analysis()
        
        self.assertEqual(result['stats'], {})
        self.assertEqual(result['trades'], [])
        self.assertEqual(result['by_symbol'], [])

    @patch('gemini_core.SessionLocal')
    def test_get_history_analysis_with_data(self, mock_session_local):
        mock_session_local.return_value = self.mock_db
        
        # Create dummy trades
        t1 = Trade(ticket=1, symbol="EURUSD", type="BUY", lot=0.1, profit=10.0, close_time=datetime.utcnow(), strategy="TREND")
        t2 = Trade(ticket=2, symbol="EURUSD", type="SELL", lot=0.1, profit=-5.0, close_time=datetime.utcnow(), strategy="REVERSION")
        
        self.mock_query.all.return_value = [t1, t2]

        result = self.gemini.get_history_analysis()
        
        self.assertEqual(result['stats']['total_profit'], 5.0)
        self.assertEqual(result['stats']['total_trades'], 2)
        self.assertEqual(result['stats']['win_rate'], 50.0)
        self.assertEqual(len(result['trades']), 2)
        self.assertEqual(len(result['by_symbol']), 1)
        self.assertEqual(result['by_symbol'][0]['symbol'], "EURUSD")
        self.assertEqual(result['by_symbol'][0]['profit'], 5.0)

    @patch('gemini_core.SessionLocal')
    def test_get_strategy_performance(self, mock_session_local):
        mock_session_local.return_value = self.mock_db
        
        # Create dummy trades
        t1 = Trade(ticket=1, profit=10.0, strategy="TREND")
        t2 = Trade(ticket=2, profit=-5.0, strategy="TREND")
        t3 = Trade(ticket=3, profit=20.0, strategy="SNIPER")
        
        self.mock_query.all.return_value = [t1, t2, t3]

        result = self.gemini.get_strategy_performance()
        
        # Sort result by name to ensure order for assertion
        result.sort(key=lambda x: x['name'])
        
        self.assertEqual(len(result), 2)
        
        # SNIPER stats
        self.assertEqual(result[0]['name'], "SNIPER")
        self.assertEqual(result[0]['trades'], 1)
        self.assertEqual(result[0]['profit'], 20.0)
        self.assertEqual(result[0]['win_rate'], 100.0)
        
        # TREND stats
        self.assertEqual(result[1]['name'], "TREND")
        self.assertEqual(result[1]['trades'], 2)
        self.assertEqual(result[1]['profit'], 5.0)
        self.assertEqual(result[1]['win_rate'], 50.0)

if __name__ == '__main__':
    unittest.main()
