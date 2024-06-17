import os
from pathlib import Path
from dotenv import load_dotenv

__version__ = "v0.0.1"

dotenv_path = Path(Path(__file__).cwd(), ".env")
load_dotenv(dotenv_path)
