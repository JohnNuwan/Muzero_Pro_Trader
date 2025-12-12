"""
V20 Constituent Analyzer
Analyse quotidienne (20h) des composants d'indices pour prédire les CFD
"""

# Indices et leurs composants
INDICES_UNIVERSE = {
    'FR40': {  # CAC 40
        'name': 'CAC 40',
        'symbols': [
            # CAC 40 Complete (40 actions)
            'MC.PA',     # LVMH
            'OR.PA',     # L'Oréal
            'SAN.PA',    # Sanofi
            'TTE.PA',    # TotalEnergies
            'AI.PA',     # Air Liquide
            'BNP.PA',    # BNP Paribas
            'ATO.PA',    # Atos
            'CS.PA',     # AXA
            'SU.PA',     # Schneider Electric
            'RMS.PA',    # Hermès
            'SGO.PA',    # Saint-Gobain
            'CAP.PA',    # Capgemini
            'VIV.PA',    # Vivendi
            'ORA.PA',    # Orange
            'KER.PA',    # Kering
            'DG.PA',     # Vinci
            'SAF.PA',    # Safran
            'EL.PA',     # EssilorLuxottica
            'RI.PA',     # Pernod Ricard
            'PUB.PA',    # Publicis
            'EN.PA',     # Bouygues
            'DSY.PA',    # Dassault Systèmes
            'AIR.PA',    # Airbus
            'ML.PA',     # Michelin
            'ALO.PA',    # Alstom
            'BN.PA',     # Danone
            'GLE.PA',    # Société Générale
            'ACA.PA',    # Crédit Agricole
            'URW.PA',    # Unibail-Rodamco-Westfield
            'STM.PA',    # STMicroelectronics
            'TEP.PA',    # Teleperformance
            'WLN.PA',    # Worldline
            'ERF.PA',    # Eurofins Scientific
            'LR.PA',     # Legrand
            'VIE.PA',    # Veolia
            'FP.PA',     # TotalEnergies (duplicate with TTE)
            'RNO.PA',    # Renault
            'FR.PA',     # Valeo
            'ENGI.PA',   # Engie
            'HO.PA',     # Thales
        ]
    },
    'GER40': {  # DAX 40
        'name': 'DAX 40',
        'symbols': [
            # DAX 40 Complete (40 actions)
            'SIE.DE',    # Siemens
            'SAP.DE',    # SAP
            'VOW3.DE',   # Volkswagen
            'ALV.DE',    # Allianz
            'BAS.DE',    # BASF
            'BAYN.DE',   # Bayer
            'BMW.DE',    # BMW
            'DAI.DE',    # Daimler (Mercedes-Benz)
            'DBK.DE',    # Deutsche Bank
            'DB1.DE',    # Deutsche Börse
            'DTE.DE',    # Deutsche Telekom
            'EOAN.DE',   # E.ON
            'FRE.DE',    # Fresenius
            'FME.DE',    # Fresenius Medical Care
            'HEI.DE',    # HeidelbergCement
            'HEN3.DE',   # Henkel
            'IFX.DE',    # Infineon
            'LIN.DE',    # Linde
            'MRK.DE',    # Merck
            'MTX.DE',    # MTU Aero Engines
            'MUV2.DE',   # Munich Re
            'RWE.DE',    # RWE
            'ADS.DE',    # Adidas
            'CON.DE',    # Continental
            'DPW.DE',    # Deutsche Post
            'DHL.DE',    # DHL (Deutsche Post)
            'HNR1.DE',   # Hannover Re
            'PUM.DE',    # Puma
            'SHL.DE',    # Siemens Healthineers
            'VNA.DE',    # Vonovia
            'ZAL.DE',    # Zalando
            'BEI.DE',    # Beiersdorf
            'SRT3.DE',   # Sartorius
            'QIA.DE',    # Qiagen
            'AIR.DE',    # Airbus (also traded in Germany)
            'PAH3.DE',   # Porsche
            'HFG.DE',    # HelloFresh
            'EVT.DE',    # Evotec
            'SY1.DE',    # Symrise
            '1COV.DE',   # Covestro
        ]
    },
    'US500': {  # S&P 500 (Top 50)
        'name': 'S&P 500',
        'symbols': [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'WMT', 'JPM', 'MA', 'PG',
            'XOM', 'HD', 'CVX', 'MRK', 'ABBV',
            'PEP', 'KO', 'COST', 'AVGO', 'ADBE',
            'TMO', 'MCD', 'ABT', 'ACN', 'CSCO',
            'NKE', 'LLY', 'TXN', 'DHR', 'NEE',
            'NFLX', 'VZ', 'PM', 'ORCL', 'CRM',
            'INTC', 'AMD', 'QCOM', 'HON', 'UPS',
            'UNP', 'IBM', 'BA', 'C', 'GS'
        ]
    },
    'US100': {  # NASDAQ 100 (Tech focus)
        'name': 'NASDAQ 100',
        'symbols': [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'AVGO', 'COST', 'ADBE',
            'PEP', 'CSCO', 'NFLX', 'CMCSA', 'TXN',
            'INTC', 'AMD', 'QCOM', 'INTU', 'AMGN',
            'HON', 'SBUX', 'GILD', 'BKNG', 'ISRG',
            'MDLZ', 'ADP', 'VRTX', 'REGN', 'LRCX',
            # ... 70 autres
        ]
    }
}

# Schedule
ANALYSIS_TIME = "20:00"  # Heure d'analyse quotidienne

if __name__ == "__main__":
    print(f"V20 Universe:")
    for idx, config in INDICES_UNIVERSE.items():
        print(f"  {idx} ({config['name']}): {len(config['symbols'])} symbols")
