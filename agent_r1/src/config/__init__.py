"""
Configuration module for Agent-R1
"""

from pathlib import Path

CONFIG_DIR = Path(__file__).parent
AGENT_TRAINER_CONFIG = CONFIG_DIR / "agent_trainer.yaml"

__all__ = ["CONFIG_DIR", "AGENT_TRAINER_CONFIG"] 