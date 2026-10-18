"""Baixa os três modelos da IA para modelos/ e confere o SHA-256 de cada um.

Roda sozinho pelo instalar.bat e pelo app na primeira vez; arquivo que já existe e confere não é baixado de novo.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

PASTA = Path(__file__).resolve().parent / "modelos"

MODELOS = {
    # YOLO26 large (Ultralytics, AGPL-3.0): acha as 80 classes do COCO
    "yolo26l.pt": ("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
                   "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26"),
    # YuNet (OpenCV Zoo, MIT): acha rostos
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    # HSEmotion EfficientNet-B0 (Savchenko, Apache-2.0): lê a expressão do rosto
    "enet_b0_8_best_vgaf.onnx": (
        "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx",
        "fa07e841fd06c7a67ee651ea4e6e4a3a2bb5695f47b37a7da50492526f59c898"),
}


def _sha256(arq: Path) -> str:
    h = hashlib.sha256()
    with open(arq, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def faltando() -> list[str]:
    return [n for n, (_, sha) in MODELOS.items()
            if not (PASTA / n).is_file() or (sha and _sha256(PASTA / n) != sha)]


def baixar(avisar=print) -> None:
    PASTA.mkdir(exist_ok=True)
    for nome in faltando():
        url, sha = MODELOS[nome]
        avisar(f"Baixando {nome}...")
        tmp = PASTA / (nome + ".parcial")
        req = urllib.request.Request(url, headers={"User-Agent": "camera-ia"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if sha and _sha256(tmp) != sha:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{nome} veio diferente do esperado (SHA-256 não confere).")
        tmp.replace(PASTA / nome)


if __name__ == "__main__":
    try:
        baixar()
    except Exception as e:  # noqa: BLE001
        print(f"Erro: {e}")
        sys.exit(1)
    print("Modelos prontos.")
