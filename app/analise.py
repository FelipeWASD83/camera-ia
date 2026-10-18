"""A IA: YOLO acha os objetos, o YuNet acha os rostos e o HSEmotion lê a expressão de cada rosto."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch  # antes do onnxruntime: é o torch que carrega as DLLs do CUDA e do cuDNN que ele usa
import onnxruntime as ort
from ultralytics import YOLO

from app import config, rotulos

_MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DESVIO = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_LADO_EMOCAO = 224  # entrada do enet_b0_8_best_vgaf
_LADO_ROSTOS = 640  # escala da busca de rostos no quadro inteiro
_LADO_RECORTE = 480  # escala da segunda busca, dentro de cada pessoa


@dataclass
class Objeto:
    caixa: tuple[int, int, int, int]  # x1, y1, x2, y2
    nome: str
    confianca: float
    classe: int


@dataclass
class Rosto:
    caixa: tuple[int, int, int, int]
    emocao: str
    confianca: float
    probs: np.ndarray = field(repr=False)


@dataclass
class Resultado:
    objetos: list[Objeto]
    rostos: list[Rosto]
    ms: float = 0.0


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


class Analisador:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.gpu = torch.cuda.is_available()
        self.yolo = YOLO(str(config.MODELOS / cfg["modelo"]))
        # o Ultralytics põe o OpenCV em 1 thread ao ser importado; o YuNet fica bem mais lento assim
        cv2.setNumThreads(8)
        self.rostos = cv2.FaceDetectorYN.create(
            str(config.MODELOS / "face_detection_yunet_2023mar.onnx"), "", (320, 320),
            float(cfg["confianca_rosto"]), 0.3, 50)
        # ONNX Runtime em vez do cv2.dnn: mesma saída, metade do tempo na CPU, e roda na GPU
        provedores = (["CUDAExecutionProvider"] if self.gpu else []) + ["CPUExecutionProvider"]
        self.emocao = ort.InferenceSession(str(config.MODELOS / "enet_b0_8_best_vgaf.onnx"), providers=provedores)
        self._entrada_emocao = self.emocao.get_inputs()[0].name
        self._tamanho_rostos = (0, 0)
        self._anteriores: list[Rosto] = []

    def aquecer(self, frame: np.ndarray) -> None:
        for _ in range(3):
            self.analisar(frame)
        self._anteriores = []

    def _objetos(self, frame) -> list[Objeto]:
        r = self.yolo.predict(frame, device=0 if self.gpu else "cpu", quantize=16 if self.gpu else None,
                              conf=float(self.cfg["confianca_minima"]), verbose=False)[0]
        saida = []
        for xyxy, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            saida.append(Objeto(tuple(int(v) for v in xyxy), rotulos.nome(r.names[int(cls)]), conf, int(cls)))
        return saida

    def _yunet(self, img, lado: int, dx: int = 0, dy: int = 0):
        """Roda o YuNet com o lado maior da imagem em `lado` px e devolve as caixas na escala original.
        O YuNet perde rosto grande demais (perto da câmera) e pequeno demais, por isso a escala é fixa."""
        h, w = img.shape[:2]
        s = lado / max(h, w)
        g = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if self._tamanho_rostos != g.shape[1::-1]:
            self.rostos.setInputSize(g.shape[1::-1])
            self._tamanho_rostos = g.shape[1::-1]
        _, achados = self.rostos.detect(g)
        caixas = []
        for f in achados if achados is not None else []:
            x, y, fw, fh = (float(v) / s for v in f[:4])
            caixas.append((int(x) + dx, int(y) + dy, int(x + fw) + dx, int(y + fh) + dy))
        return caixas

    def _caixas_rostos(self, frame, pessoas=()) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        caixas = self._yunet(frame, _LADO_ROSTOS)
        # rosto longe da câmera some na escala de 640: procura de novo dentro de cada pessoa sem rosto
        for x1, y1, x2, y2 in pessoas:
            if any(x1 <= (c[0] + c[2]) / 2 <= x2 and y1 <= (c[1] + c[3]) / 2 <= y2 for c in caixas):
                continue
            alto = y1 + int((y2 - y1) * 0.6)
            recorte = frame[y1:alto, x1:x2]
            if recorte.shape[0] < 16 or recorte.shape[1] < 16:
                continue
            for c in self._yunet(recorte, _LADO_RECORTE, x1, y1):
                if all(_iou(c, d) < 0.3 for d in caixas):
                    caixas.append(c)
        saida = []
        for x1, y1, x2, y2 in caixas:
            # margem de 10%: o HSEmotion foi treinado com o rosto um pouco folgado e erra menos assim
            mx, my = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
            x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
            if x2 - x1 >= 16 and y2 - y1 >= 16:
                saida.append((x1, y1, x2, y2))
        return saida

    def _emocoes(self, frame, caixas) -> list[np.ndarray]:
        if not caixas:
            return []
        lote = []
        for x1, y1, x2, y2 in caixas:
            rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (_LADO_EMOCAO, _LADO_EMOCAO), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            lote.append(((img - _MEDIA) / _DESVIO).transpose(2, 0, 1))
        logits = self.emocao.run(None, {self._entrada_emocao: np.stack(lote)})[0].astype(np.float64)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return list(e / e.sum(axis=1, keepdims=True))

    def _suavizar(self, caixas, probs) -> list[Rosto]:
        """Média com o mesmo rosto no quadro anterior, para a expressão não piscar de um quadro para o outro."""
        rostos = []
        for caixa, p in zip(caixas, probs):
            par = max(self._anteriores, key=lambda r: _iou(r.caixa, caixa), default=None)
            if par is not None and _iou(par.caixa, caixa) > 0.3:
                p = 0.55 * p + 0.45 * par.probs
            i = int(np.argmax(p))
            rostos.append(Rosto(caixa, rotulos.EMOCOES[i], float(p[i]), p))
        self._anteriores = rostos
        return rostos

    def analisar(self, frame: np.ndarray) -> Resultado:
        t = time.perf_counter()
        objetos = self._objetos(frame)
        caixas = self._caixas_rostos(frame, [o.caixa for o in objetos if o.classe == 0])
        rostos = self._suavizar(caixas, self._emocoes(frame, caixas))
        return Resultado(objetos, rostos, (time.perf_counter() - t) * 1000)
