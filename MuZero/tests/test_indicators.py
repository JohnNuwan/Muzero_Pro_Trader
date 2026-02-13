import unittest
import pandas as pd
import numpy as np
from MuZero.utils.indicators import Indicators

class TestIndicators(unittest.TestCase):
    def setUp(self):
        # Create a synthetic DataFrame
        np.random.seed(42)
        n = 500  # Increased to ensure sufficient data after dropna()
        dates = pd.date_range(start='2023-01-01', periods=n, freq='h')

        # Simulate price movement
        close = np.cumsum(np.random.randn(n)) + 100
        high = close + np.random.rand(n) * 2
        low = close - np.random.rand(n) * 2
        open_p = close + np.random.randn(n) * 0.5

        # Ensure high is highest and low is lowest
        high = np.maximum(high, np.maximum(open_p, close))
        low = np.minimum(low, np.minimum(open_p, close))

        self.df = pd.DataFrame({
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'tick_volume': np.random.randint(100, 1000, n)
        }, index=dates)

    def test_rsi(self):
        df = Indicators.rsi(self.df.copy(), period=14)
        self.assertIn('rsi', df.columns)
        # RSI should be between 0 and 100
        # dropna because first 14 will be NaN
        rsi = df['rsi'].dropna()
        self.assertTrue(((rsi >= 0) & (rsi <= 100)).all())

    def test_mfi(self):
        df = Indicators.mfi(self.df.copy(), period=14)
        self.assertIn('mfi', df.columns)
        mfi = df['mfi'].dropna()
        self.assertTrue(((mfi >= 0) & (mfi <= 100)).all())

    def test_ema(self):
        df = Indicators.ema(self.df.copy(), period=20)
        self.assertIn('ema_20', df.columns)
        self.assertEqual(len(df), len(self.df))

    def test_atr(self):
        df = Indicators.atr(self.df.copy(), period=14)
        self.assertIn('atr', df.columns)
        self.assertTrue((df['atr'].dropna() > 0).all())

    def test_adx(self):
        df = Indicators.adx(self.df.copy(), period=14)
        self.assertIn('adx', df.columns)
        # ADX logic check? ADX >= 0
        self.assertTrue((df['adx'].dropna() >= 0).all())

    def test_obv(self):
        df = Indicators.obv(self.df.copy())
        self.assertIn('obv', df.columns)
        # Check first value is 0 (fillna(0) then cumsum, but first diff is NaN which becomes 0)
        # Actually first diff is NaN, sign is NaN, fillna(0) makes it 0. So OBV starts at 0.
        self.assertEqual(df['obv'].iloc[0], 0)

    def test_z_score(self):
        df = Indicators.z_score(self.df.copy(), period=20)
        self.assertIn('z_score', df.columns)
        # Z-score can be negative or positive

    def test_linear_regression(self):
        df = Indicators.linear_regression(self.df.copy(), period=20)
        self.assertIn('linreg_slope', df.columns)
        self.assertIn('linreg_angle', df.columns)

    def test_pivots(self):
        df = Indicators.pivots(self.df.copy())
        self.assertIn('pivot', df.columns)
        self.assertIn('r1', df.columns)
        self.assertIn('s1', df.columns)
        self.assertIn('r2', df.columns)
        self.assertIn('s2', df.columns)

    def test_fibonacci(self):
        df = Indicators.fibonacci(self.df.copy(), period=100)
        self.assertIn('fibo_0', df.columns)
        self.assertIn('fibo_100', df.columns)
        self.assertIn('fibo_pos', df.columns)

    def test_support_resistance(self):
        df = Indicators.support_resistance(self.df.copy(), period=20)
        self.assertIn('resistance', df.columns)
        self.assertIn('support', df.columns)
        self.assertIn('dist_to_res', df.columns)
        self.assertIn('dist_to_sup', df.columns)

    def test_accumulation_distribution(self):
        df = Indicators.accumulation_distribution(self.df.copy())
        self.assertIn('ad_line', df.columns)

    def test_statistical_moments(self):
        df = Indicators.statistical_moments(self.df.copy(), period=20)
        self.assertIn('skew', df.columns)
        self.assertIn('kurtosis', df.columns)

    def test_shannon_entropy(self):
        df = Indicators.shannon_entropy(self.df.copy())
        self.assertIn('entropy', df.columns)
        # Currently a placeholder returning constant 0.5
        self.assertTrue((df['entropy'] == 0.5).all())

    def test_hurst_exponent(self):
        df = Indicators.hurst_exponent(self.df.copy())
        self.assertIn('hurst', df.columns)
        # Currently a placeholder returning constant 0.5
        self.assertTrue((df['hurst'] == 0.5).all())

    def test_kalman_filter(self):
        df = Indicators.kalman_filter(self.df.copy())
        self.assertIn('kalman_price', df.columns)
        self.assertIn('kalman_diff', df.columns)

    def test_trend_strength(self):
        # Test without existing EMAs
        df = Indicators.trend_strength(self.df.copy())
        self.assertIn('trend_score', df.columns)
        self.assertIn('ema_20', df.columns)
        self.assertIn('ema_50', df.columns)
        self.assertIn('ema_200', df.columns)

        # Trend score should be -1, 0, or 1
        self.assertTrue(df['trend_score'].isin([-1.0, 0.0, 1.0]).all())

        # Test with existing EMAs
        df2 = self.df.copy()
        df2['ema_20'] = df2['close'] # Fake EMAs
        df2['ema_50'] = df2['close']
        df2['ema_200'] = df2['close']
        df2 = Indicators.trend_strength(df2)
        # Should rely on existing columns
        # If all are equal, conditions are False (using > and <), so default 0.0
        self.assertTrue((df2['trend_score'] == 0.0).all())

    def test_stochastic_rsi(self):
        df = Indicators.stochastic_rsi(self.df.copy())
        self.assertIn('stoch_rsi_k', df.columns)
        self.assertIn('stoch_rsi_d', df.columns)
        # Should be between 0 and 1 (approx, moving average might smooth it, but source is 0-1)
        # Actually stochastic is 0-1.

    def test_williams_r(self):
        df = Indicators.williams_r(self.df.copy())
        self.assertIn('williams_r', df.columns)
        # Should be between -100 and 0
        williams = df['williams_r'].dropna()
        self.assertTrue(((williams >= -100) & (williams <= 0)).all())

    def test_cci(self):
        df = Indicators.cci(self.df.copy())
        self.assertIn('cci', df.columns)

    def test_bollinger_bands(self):
        df = Indicators.bollinger_bands(self.df.copy())
        self.assertIn('bb_upper', df.columns)
        self.assertIn('bb_lower', df.columns)
        self.assertIn('bb_middle', df.columns)
        self.assertIn('bb_width', df.columns)
        self.assertIn('bb_percent_b', df.columns)
        # Upper should be >= Lower
        df = df.dropna()
        self.assertTrue((df['bb_upper'] >= df['bb_lower']).all())

    def test_keltner_channels(self):
        df = Indicators.keltner_channels(self.df.copy())
        self.assertIn('keltner_upper', df.columns)
        self.assertIn('keltner_lower', df.columns)
        self.assertIn('keltner_middle', df.columns)
        df = df.dropna()
        self.assertTrue((df['keltner_upper'] >= df['keltner_lower']).all())

    def test_atr_bands(self):
        df = Indicators.atr_bands(self.df.copy())
        self.assertIn('atr_upper', df.columns)
        self.assertIn('atr_lower', df.columns)
        df = df.dropna()
        self.assertTrue((df['atr_upper'] >= df['atr_lower']).all())

    def test_vwap(self):
        df = Indicators.vwap(self.df.copy())
        self.assertIn('vwap', df.columns)
        self.assertIn('vwap_signal', df.columns)
        self.assertTrue(df['vwap_signal'].isin([-1, 1]).all())

    def test_volume_profile(self):
        df = Indicators.volume_profile(self.df.copy())
        self.assertIn('volume_concentration', df.columns)
        self.assertIn('price_bin', df.columns)
        # Concentration should be float
        self.assertIsInstance(df['volume_concentration'].iloc[0], float)

    def test_add_all(self):
        df = Indicators.add_all(self.df.copy())
        # Check that we have a lot of columns
        initial_cols = set(self.df.columns)
        final_cols = set(df.columns)
        self.assertTrue(len(final_cols) > len(initial_cols))

        # Check no NaNs and not empty
        self.assertFalse(df.isna().any().any())
        self.assertTrue(len(df) > 0, "Resulting DataFrame should not be empty with sufficient input data")

    def test_edge_cases(self):
        # Empty DataFrame
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'tick_volume'])

        # RSI on empty df should return empty df with 'rsi' column
        df = Indicators.rsi(empty_df.copy())
        self.assertTrue(df.empty)
        self.assertIn('rsi', df.columns)

        # Single row
        single_row_df = self.df.iloc[:1].copy()
        # Most indicators need history, so they will produce NaNs
        # add_all drops NaNs, so it should return empty DataFrame
        df = Indicators.add_all(single_row_df.copy())
        self.assertTrue(df.empty)

if __name__ == '__main__':
    unittest.main()
