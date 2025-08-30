#!/usr/bin/env python3
"""
Simple logger setup for FusionAlpha routers.
"""

import logging
import os
import sys
from pathlib import Path


def setup_router_logging():
    """Setup basic logging for router modules."""
    # Get log level from environment
    level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Create logs directory
    log_dir = Path(__file__).parent.parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "routers.log")
        ]
    )


# Auto-setup when module is imported
if not logging.getLogger().handlers:
    setup_router_logging()