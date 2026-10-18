"""Desenha as caixas, os rótulos e o painel em cima do quadro da câmera.

As caixas saem pelo OpenCV, direto no quadro. O texto sai pelo Pillow com a Segoe UI (as fontes do
OpenCV não têm acento), mas só o pedacinho de cada rótulo, guardado em cache e colado no quadro:
desenhar o quadro inteiro no Pillow custava 33 ms em 1080p, mais que o tempo de um quadro da câmera.
Tudo aqui é BGR, como o quadro do OpenCV.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import rotulos

_FONTE = r"C:\Windows\Fonts\seguisb.ttf"
_FONTE_RESERVA = r"C:\Windows\Fonts\segoeui.ttf"

# Cores das caixas dos objetos (BGR), uma por classe do COCO, em rodízio.
_PALETA = [
    (232, 162, 0), (0, 196, 255), (80, 200, 60), (210, 90, 230), (60, 110, 255), (200, 200, 40),
    (255, 120, 120), (40, 230, 200), (150, 80, 255), (90, 170, 255), (255, 200, 90), (120, 255, 160),
]
_COR_PESSOA = (235, 150, 20)


@lru_cache(maxsize=16)
def fonte(tamanho: int) -> ImageFont.FreeTypeFont:
    for arq in (_FONTE, _FONTE_RESERVA):
        try:
            return ImageFont.truetype(arq, tamanho)
        except OSError:
            continue
    return ImageFont.load_default(tamanho)


def cor_objeto(classe: int) -> tuple[int, int, int]:
    return _COR_PESSOA if classe == 0 else _PALETA[classe % len(_PALETA)]


def _texto_sobre(cor) -> tuple[int, int, int]:
    b, g, r = cor[:3]
    return (20, 20, 20) if 0.299 * r + 0.587 * g + 0.114 * b > 140 else (255, 255, 255)


@lru_cache(maxsize=1024)
def etiqueta(texto: str, tamanho: int, fundo: tuple, cor_texto: tuple | None = None) -> np.ndarray:
    """Texto num retângulo arredondado, como BGRA (o canto arredondado é transparente)."""
    f = fonte(tamanho)
    esq, topo, dir_, base = f.getbbox(texto)
    pad = max(4, round(tamanho * 0.3))
    lw, lh = dir_ - esq + 2 * pad, base - topo + 2 * pad
    alfa = fundo[3] if len(fundo) == 4 else 255
    img = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, lw - 1, lh - 1), radius=max(3, round(tamanho * 0.22)), fill=(*fundo[:3], alfa))
    d.text((pad - esq, pad - topo), texto, font=f, fill=(*(cor_texto or _texto_sobre(fundo)), 255))
    return np.asarray(img)


def colar(frame: np.ndarray, peca: np.ndarray, x: int, y: int) -> None:
    """Cola a peça BGRA no quadro BGR, misturando pela transparência; corta o que sair da imagem."""
    h, w = frame.shape[:2]
    ph, pw = peca.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + pw), min(h, y + ph)
    if x1 <= x0 or y1 <= y0:
        return
    p = peca[y0 - y:y1 - y, x0 - x:x1 - x]
    a = p[..., 3:4].astype(np.uint16)
    alvo = frame[y0:y1, x0:x1]
    alvo[:] = ((p[..., :3].astype(np.uint16) * a + alvo.astype(np.uint16) * (255 - a)) // 255).astype(np.uint8)


def _caixa_com_rotulo(frame, caixa, texto: str, cor, escala: float, embaixo=False):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = caixa
    traco = max(2, round(3 * escala))
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, traco, cv2.LINE_AA)
    peca = etiqueta(texto, max(12, round(24 * escala)), tuple(cor))
    ph, pw = peca.shape[:2]
    lx = min(max(0, x1 - traco // 2), max(0, w - pw))
    if embaixo and y2 + ph <= h:
        ly = y2
    else:
        ly = y1 - ph if y1 - ph >= 0 else y1  # sem espaço em cima, o rótulo entra na caixa
    colar(frame, peca, lx, ly)


def desenhar(frame: np.ndarray, resultado) -> np.ndarray:
    """Devolve uma cópia do quadro com as caixas dos objetos e dos rostos. `resultado` None = IA carregando."""
    img = frame.copy()
    if resultado is None:
        return img
    escala = img.shape[0] / 1080
    # maiores primeiro: assim o rótulo de um objeto pequeno não fica escondido atrás da caixa de um grande
    for o in sorted(resultado.objetos, key=lambda o: -(o.caixa[2] - o.caixa[0]) * (o.caixa[3] - o.caixa[1])):
        _caixa_com_rotulo(img, o.caixa, f"{o.nome} {o.confianca * 100:.0f}%", cor_objeto(o.classe), escala)
    # a expressão vai embaixo do rosto: em cima ela cobria o rótulo "Pessoa", que fica logo acima da cabeça
    for r in resultado.rostos:
        _caixa_com_rotulo(img, r.caixa, f"{r.emocao} {r.confianca * 100:.0f}%", rotulos.COR_EMOCAO[r.emocao],
                          escala * 0.9, embaixo=True)
    return img


def painel(frame: np.ndarray, info: list[str], teclas: str, aviso: str | None) -> np.ndarray:
    """Informação no canto de baixo à esquerda, teclas embaixo à direita, aviso no meio de cima. Desenha no próprio quadro."""
    h, w = frame.shape[:2]
    escala = h / 1080
    m = round(14 * escala)
    tam = max(11, round(19 * escala))
    fundo = (0, 0, 0, 160)
    y = h - m
    for linha in reversed(info):
        peca = etiqueta(linha, tam, fundo, (255, 255, 255))
        y -= peca.shape[0]
        colar(frame, peca, m, y)
        y -= round(6 * escala)
    if teclas:
        peca = etiqueta(teclas, tam, fundo, (255, 255, 255))
        colar(frame, peca, w - m - peca.shape[1], h - m - peca.shape[0])
    if aviso:
        peca = etiqueta(aviso, max(12, round(24 * escala)), (30, 200, 255, 235), (20, 20, 20))
        colar(frame, peca, (w - peca.shape[1]) // 2, m)
    return frame


def tela_de_espera(largura: int, altura: int, texto: str) -> np.ndarray:
    img = np.full((altura, largura, 3), (20, 22, 24), np.uint8)
    peca = etiqueta(texto, max(14, round(34 * altura / 1080)), (20, 22, 24), (235, 235, 235))
    colar(img, peca, (largura - peca.shape[1]) // 2, (altura - peca.shape[0]) // 2)
    return img
