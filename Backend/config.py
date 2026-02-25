import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repository root
ROOT = Path(__file__).resolve().parents[1]
dotenv_path = ROOT / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# Defaults
DATABASE_PATH = os.getenv('DATABASE_PATH') or str(Path(__file__).resolve().parents[0] / 'data.db')
STORAGE_DIR = os.getenv('STORAGE_DIR') or str(Path(__file__).resolve().parents[0] / 'storage')
SECRET_KEY = os.getenv('SECRET_KEY') or 'dataset-cleaner-secret-key'
