"""Lê a webcam numa thread própria e guarda sempre o quadro mais novo."""
from __future__ import annotations

import os
import threading
import time

# Sem isto o OpenCV leva quase 1 minuto para abrir a C920 pelo Media Foundation (medido: 56 s contra 1,6 s).
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import cv2  # noqa: E402


class Camera(threading.Thread):
    def __init__(self, indice: int, largura: int, altura: int):
        super().__init__(daemon=True, name="camera")
        self.indice, self.largura, self.altura = indice, largura, altura
        self._cond = threading.Condition()
        self._quadro = None
        self.numero = 0
        self.conectada = False
        self.erro = ""
        self._parar = threading.Event()

    def _abrir(self):
        cap = cv2.VideoCapture(self.indice, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.altura)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def run(self):
        while not self._parar.is_set():
            cap = self._abrir()
            if not cap.isOpened():
                self.conectada, self.erro = False, "Não achei a câmera. Ela está conectada e livre?"
                cap.release()
                time.sleep(2)
                continue
            falhas = 0
            while not self._parar.is_set():
                ok, quadro = cap.read()
                if not ok:
                    falhas += 1
                    if falhas > 30:  # outro programa pegou a câmera ou ela foi desconectada
                        self.conectada, self.erro = False, "A câmera parou de mandar imagem. Tentando de novo..."
                        break
                    time.sleep(0.03)
                    continue
                falhas = 0
                with self._cond:
                    self._quadro = quadro
                    self.numero += 1
                    self.conectada, self.erro = True, ""
                    self._cond.notify_all()
            cap.release()

    def esperar(self, depois_de: int, tempo: float = 0.1):
        """Espera um quadro mais novo que `depois_de`. Devolve (quadro, numero); quadro None se não chegou."""
        with self._cond:
            if self.numero <= depois_de:
                self._cond.wait(tempo)
            if self.numero <= depois_de or self._quadro is None:
                return None, self.numero
            return self._quadro, self.numero

    def parar(self):
        self._parar.set()
