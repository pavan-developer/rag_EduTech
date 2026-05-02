"""
@module    conftest
@description Pytest configuration — adds project root to Python path
             so all modules can be imported correctly during testing.
@author    EduMind AI Engineering
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath("."))