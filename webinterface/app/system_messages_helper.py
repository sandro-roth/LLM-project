# webinterface/app/system_messages_helper.py
from pathlib import Path
from jinja2 import Template
import yaml

def _find_yaml(yaml_path: str | None = None) -> Path:
    if yaml_path:
        return Path(yaml_path).resolve()

    here = Path(__file__).resolve().parent
    candidates = [
        here / "system_messages.yml",
        here.parent / "system_messages.yml",         # falls aus pages/ aufgerufen
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("system_messages.yml nicht gefunden (getestete Pfade: {})"
                            .format([str(c) for c in candidates]))

def load_messages(yaml_path: str | None = None) -> dict:
    path = _find_yaml(yaml_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def render_system_message(key: str, overrides: dict | None = None, yaml_path: str | None = None) -> str:
    data = load_messages(yaml_path)
    entry = data.get(key) or {}
    template_str = (overrides or {}).get(key) or entry.get("template", "")
    context = entry.get("context", {}) or {}
    return Template(template_str).render(context)
