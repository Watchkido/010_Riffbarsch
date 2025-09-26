"""
logging_config.py
Konfiguration für das Logging-System.
Hier wird das Logging-Format und Level festgelegt.
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
