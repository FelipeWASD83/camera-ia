"""Janela do Câmera IA: mostra a webcam com as caixas da IA por cima.

Três threads: a câmera lê sem parar, a IA analisa o quadro mais novo quando termina o anterior,
e esta aqui desenha cada quadro que chega com o último resultado da IA. Assim o vídeo fica nos 30 fps
da câmera mesmo se a IA demorar mais que um quadro.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
import traceback
from collections import deque

from app.camera import Camera, cv2  # a câmera ajusta o ambiente do OpenCV antes do import
from app import atualizador, config, desenho


# O OpenCV escreve o título da janela em ANSI ("CÃ¢mera IA"): a janela nasce com um nome sem acento,
# que é o que as funções do OpenCV usam, e o título de verdade é trocado pela API do Windows.
JANELA = "camera-ia"
TITULO = "Câmera IA"
TECLAS = "F tela cheia  ·  U procurar atualização  ·  Q sair"


def _ajustar_janela() -> None:
    """Título com acento, o ícone do app e um id próprio na barra de tarefas (senão ela agrupa com o Python)."""
    try:
        from ctypes import c_int, c_void_p, windll, wintypes
        user32 = windll.user32
        user32.FindWindowW.restype = wintypes.HWND
        user32.LoadImageW.restype = c_void_p
        user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, c_int, c_int,
                                      wintypes.UINT]
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        hwnd = user32.FindWindowW(None, JANELA)
        if not hwnd:
            return
        user32.SetWindowTextW(hwnd, TITULO)
        ico = str(config.RAIZ / "icone" / "icone.ico")
        for tipo, lado in ((1, 48), (0, 16)):  # ICON_BIG, ICON_SMALL
            h = user32.LoadImageW(None, ico, 1, lado, lado, 0x10)  # IMAGE_ICON, LR_LOADFROMFILE
            if h:
                user32.SendMessageW(hwnd, 0x80, tipo, h)  # WM_SETICON
    except (OSError, AttributeError):
        pass


class Medidor:
    """Quadros por segundo nos últimos 2 s."""

    def __init__(self):
        self.marcas = deque()

    def marcar(self):
        agora = time.perf_counter()
        self.marcas.append(agora)
        while self.marcas and agora - self.marcas[0] > 2:
            self.marcas.popleft()

    def fps(self) -> float:
        if len(self.marcas) < 2:
            return 0.0
        return (len(self.marcas) - 1) / (self.marcas[-1] - self.marcas[0])


class IA(threading.Thread):
    def __init__(self, cfg, camera: Camera):
        super().__init__(daemon=True, name="ia")
        self.cfg, self.camera = cfg, camera
        self.resultado = None
        self.estado = "Preparando a IA..."
        self.erro = ""
        self.medidor = Medidor()
        self.parar = threading.Event()

    def run(self):
        try:
            import baixar_modelos
            if baixar_modelos.faltando():
                baixar_modelos.baixar(lambda m: setattr(self, "estado", m))
            self.estado = "Carregando a IA na placa de vídeo..."
            from app.analise import Analisador  # importa o torch: demora alguns segundos
            analisador = Analisador(self.cfg)
            visto = 0
            aquecido = False
            while not self.parar.is_set():
                quadro, n = self.camera.esperar(visto, 0.2)
                if quadro is None:
                    continue
                visto = n
                if not aquecido:
                    analisador.aquecer(quadro)
                    aquecido = True
                    self.estado = ""
                self.resultado = analisador.analisar(quadro)
                self.medidor.marcar()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.erro = f"A IA não carregou: {e}"


class Atualizacao:
    """Procura versão nova em segundo plano e, com A, baixa e pede ao iniciar.pyw para trocar."""

    def __init__(self):
        self.info: dict | None = None
        self.mensagem = ""
        self.mensagem_ate = 0.0
        self.ocupado = False
        self.pronta = False

    def avisar(self, texto: str, segundos: float = 6):
        self.mensagem, self.mensagem_ate = texto, time.time() + segundos

    def procurar(self, forcar=False, avisar=False):
        if self.ocupado:
            return
        self.ocupado = True
        if avisar:
            self.avisar("Procurando atualização...", 30)

        def tarefa():
            try:
                self.info = atualizador.checar(forcar)
                if not avisar:
                    return
                if self.info.get("erro"):
                    self.avisar(self.info["erro"], 8)
                elif self.info.get("nova"):
                    self.avisar("", 0)
                else:
                    self.avisar(f"Você já está na versão mais nova ({config.VERSAO}).")
            finally:
                self.ocupado = False

        threading.Thread(target=tarefa, daemon=True).start()

    def instalar(self):
        if self.ocupado or not (self.info and self.info.get("nova")):
            return
        self.ocupado = True
        self.avisar(f"Baixando a versão {self.info['ultima']}...", 600)

        def tarefa():
            try:
                atualizador.baixar_e_preparar()
                self.pronta = True
            except atualizador.ErroAtualizacao as e:
                self.avisar(str(e), 10)
            except Exception as e:  # noqa: BLE001
                self.avisar(f"Não deu para baixar: {e}", 10)
            finally:
                self.ocupado = False

        threading.Thread(target=tarefa, daemon=True).start()

    def faixa(self) -> str | None:
        if time.time() < self.mensagem_ate and self.mensagem:
            return self.mensagem
        if self.info and self.info.get("nova"):
            return f"Versão {self.info['ultima']} disponível: aperte A para atualizar"
        return None


def janela_aberta() -> bool:
    try:
        return cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def _argumentos():
    p = argparse.ArgumentParser(description="Câmera IA")
    p.add_argument("--sair-em", type=float, default=0, help="teste: fecha sozinho depois de N segundos de vídeo")
    p.add_argument("--salvar-quadro", default="", help="teste: salva o último quadro mostrado neste arquivo")
    return p.parse_args()


def main() -> int:
    args = _argumentos()
    cfg = config.carregar()
    camera = Camera(int(cfg["camera"]), int(cfg["largura"]), int(cfg["altura"]))
    camera.start()
    ia = IA(cfg, camera)
    ia.start()
    atualizacao = Atualizacao()
    atualizacao.procurar()

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("FelipeWASD83.CameraIA")
    except (OSError, AttributeError):
        pass
    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(JANELA, 1600, 900)
    _ajustar_janela()
    tela_cheia = False
    video = Medidor()
    visto = 0
    marcou_abriu = False
    codigo = 0
    inicio_video = None
    amostras = []  # teste: (fps do vídeo, fps da IA, ms da IA) uma vez por segundo, depois que a IA aquece
    ultima_amostra = 0.0
    img = None

    while True:
        quadro, n = camera.esperar(visto, 0.05)
        if quadro is not None:
            if args.sair_em and ia.resultado is not None:
                agora = time.perf_counter()
                inicio_video = inicio_video or agora
                if agora - ultima_amostra >= 1 and agora - inicio_video > 3:
                    amostras.append((video.fps(), ia.medidor.fps(), ia.resultado.ms))
                    ultima_amostra = agora
                if agora - inicio_video > args.sair_em:
                    break
            visto = n
            video.marcar()
            img = desenho.desenhar(quadro, ia.resultado)
            linhas = []
            if cfg.get("mostrar_fps", True):
                r = ia.resultado
                linhas.append(f"Vídeo {video.fps():.0f} fps  ·  IA {ia.medidor.fps():.0f} fps"
                              + (f" ({r.ms:.0f} ms)" if r else ""))
            if ia.erro or ia.estado:
                linhas.append(ia.erro or ia.estado)
            img = desenho.painel(img, linhas, TECLAS, atualizacao.faixa())
            cv2.imshow(JANELA, img)
            # só conta como "abriu" quando a IA respondeu: versão nova com a IA quebrada também volta para a anterior
            if not marcou_abriu and ia.resultado is not None:
                atualizador.PASTA.mkdir(parents=True, exist_ok=True)
                atualizador.ABRIU.touch()
                marcou_abriu = True
        elif not camera.conectada:
            texto = camera.erro or "Abrindo a câmera..."
            cv2.imshow(JANELA, desenho.painel(desenho.tela_de_espera(1600, 900, texto), [], TECLAS,
                                              atualizacao.faixa()))

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), ord("Q"), 27) or not janela_aberta():
            break
        if tecla in (ord("f"), ord("F")):
            tela_cheia = not tela_cheia
            cv2.setWindowProperty(JANELA, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if tela_cheia else cv2.WINDOW_NORMAL)
        if tecla in (ord("u"), ord("U")):
            atualizacao.procurar(forcar=True, avisar=True)
        if tecla in (ord("a"), ord("A")):
            atualizacao.instalar()
        if atualizacao.pronta:
            codigo = atualizador.SAIR_PARA_ATUALIZAR
            break

    ia.parar.set()
    camera.parar()
    cv2.destroyAllWindows()
    if args.salvar_quadro and img is not None:
        cv2.imwrite(args.salvar_quadro, img)
    if amostras:
        def mediana(i):
            return sorted(a[i] for a in amostras)[len(amostras) // 2]
        print(f"mediana em {len(amostras)} s: vídeo {mediana(0):.1f} fps, IA {mediana(1):.1f} fps, "
              f"IA {mediana(2):.1f} ms por quadro")
    if ia.erro:
        print(ia.erro)
        return codigo or 1
    return codigo


if __name__ == "__main__":
    sys.exit(main())
