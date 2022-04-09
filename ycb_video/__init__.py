import json
import os


def _load_config():
    with open(f"{os.path.dirname(__file__)}/config.json") as f:
        json_str = f.read()
        return json.loads(json_str)


CONFIG = _load_config()
