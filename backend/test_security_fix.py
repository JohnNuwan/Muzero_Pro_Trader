import sys
from unittest.mock import MagicMock

# Mock MetaTrader5 before importing gemini_core
sys.modules["MetaTrader5"] = MagicMock()

import unittest
from unittest.mock import patch, mock_open
import json
import os
from gemini_core import GeminiCore

class TestGeminiSecurity(unittest.TestCase):
    def setUp(self):
        # Mock SessionLocal to prevent DB connection
        self.session_patcher = patch('gemini_core.SessionLocal')
        self.mock_session = self.session_patcher.start()

        # Mock os.makedirs to prevent folder creation
        self.makedirs_patcher = patch('os.makedirs')
        self.makedirs_patcher.start()

        # Mock open to prevent reading/writing config file during init
        with patch("builtins.open", mock_open(read_data='{"EURUSD": {"lot": 0.1}}')):
            self.gemini = GeminiCore()

        # Inject sensitive data into params to simulate a leak scenario
        self.gemini.params["EURUSD"] = {
            "lot": 0.1,
            "password": "LEAKED_PASSWORD",
            "api_key": "LEAKED_KEY"
        }

    def tearDown(self):
        self.session_patcher.stop()
        self.makedirs_patcher.stop()

    def test_get_config_filters_sensitive_data(self):
        """Test that get_config() removes sensitive keys"""
        config = self.gemini.get_config()
        symbols = config.get("symbols", {})
        eurusd = symbols.get("EURUSD", {})

        self.assertIn("lot", eurusd)
        self.assertNotIn("password", eurusd)
        self.assertNotIn("api_key", eurusd)

    @patch("builtins.open", new_callable=mock_open)
    def test_update_config_blocks_sensitive_keys(self, mock_file):
        """Test that update_config() rejects sensitive keys"""
        # 1. Safe update
        safe_update = {"symbols": {"EURUSD": {"new_lot": 0.2}}}
        result = self.gemini.update_config(safe_update)
        self.assertTrue(result)
        self.assertEqual(self.gemini.params["EURUSD"]["new_lot"], 0.2)

        # 2. Sensitive update (should be blocked)
        sensitive_update = {"symbols": {"EURUSD": {"secret_token": "HACKED"}}}
        result = self.gemini.update_config(sensitive_update)
        self.assertFalse(result)
        self.assertNotIn("secret_token", self.gemini.params["EURUSD"])

    @patch("builtins.open", new_callable=mock_open)
    def test_update_config_blocks_nested_sensitive_keys(self, mock_file):
        """Test that update_config() rejects nested sensitive keys"""
        sensitive_update = {"symbols": {"EURUSD": {"settings": {"my_password": "123"}}}}
        result = self.gemini.update_config(sensitive_update)
        self.assertFalse(result)
        self.assertNotIn("settings", self.gemini.params["EURUSD"])

if __name__ == '__main__':
    unittest.main()
