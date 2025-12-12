"""
V19 Pyramiding Configuration
"""

PYRAMID_CONFIG = {
    'enabled': True,
    'max_pyramids': 3,             # Maximum number of pyramid positions
    'pyramid_volume_ratio': 0.5,   # Ratio of main position volume (0.5 = 50%)
    'min_confidence': 0.60,        # Minimum MCTS confidence required to pyramid
    'sl_trigger_profit': 0.0010,   # Profit % required to move SL to BE (0.10%)
    'min_main_profit': 0.0,        # Main position must be at least break-even
}
