import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "sources.yaml"

def load_sources():
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data["sources"]