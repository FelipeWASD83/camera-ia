"""Caminhos, versão e preferências do Câmera IA."""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
MODELOS = RAIZ / "modelos"

PADRAO = {
    "camera": 0,
    "largura": 1920,
    "altura": 1080,
    "modelo": "yolo26l.pt",
    "confianca_minima": 0.40,
    "confianca_rosto": 0.75,
    "mostrar_fps": True,
}


def info_versao() -> dict:
    try:
        return json.loads((RAIZ / "versao.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


VERSAO = info_versao().get("versao", "0.0.0")


def carregar() -> dict:
    """Lê dados/config.json por cima do padrão; cria o arquivo na primeira vez para ele poder editar."""
    DADOS.mkdir(parents=True, exist_ok=True)
    arq = DADOS / "config.json"
    cfg = dict(PADRAO)
    try:
        cfg.update(json.loads(arq.read_text(encoding="utf-8")))
    except FileNotFoundError:
        arq.write_text(json.dumps(PADRAO, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass
    return cfg
