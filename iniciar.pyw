"""Abre o Câmera IA (é o que o atalho do Menu Iniciar chama, pelo pythonw do .venv).

Também aplica a atualização baixada pelo app e volta para a versão anterior se a nova não abrir.
Erros vão para dados/log.txt, já que o pythonw não tem console.
"""
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from app import atualizador, config  # noqa: E402

LOG = config.DADOS / "log.txt"
SEM_JANELA = 0x08000000  # CREATE_NO_WINDOW


def _python_com_console() -> str:
    exe = Path(sys.executable)
    return str(exe.with_name("python.exe")) if exe.name.lower() == "pythonw.exe" else str(exe)


def _instalar_dependencias():
    with open(LOG, "a", encoding="utf-8") as log:
        subprocess.call([_python_com_console(), "-m", "pip", "install", "-r", str(RAIZ / "requirements.txt")],
                        cwd=RAIZ, stdout=log, stderr=log, creationflags=SEM_JANELA)


def _aplicar():
    try:
        info = atualizador.aplicar_pendente(RAIZ)
    except atualizador.ErroAtualizacao as e:
        with open(LOG, "a", encoding="utf-8") as log:
            log.write(f"Atualização não aplicada: {e}\n")
        return None
    if info and info["dependencias_mudaram"]:
        _instalar_dependencias()
    return info


def _rodar() -> int:
    with open(LOG, "w", encoding="utf-8") as log:
        return subprocess.call([sys.executable, "-m", "app.principal"], cwd=RAIZ, stdout=log, stderr=log)


def main():
    config.DADOS.mkdir(parents=True, exist_ok=True)
    acabou_de_atualizar = _aplicar()
    while True:
        atualizador.ABRIU.unlink(missing_ok=True)
        codigo = _rodar()
        if codigo == atualizador.SAIR_PARA_ATUALIZAR:
            acabou_de_atualizar = _aplicar()
            continue
        if codigo != 0 and acabou_de_atualizar and not atualizador.ABRIU.exists():
            # a versão nova nem chegou a mostrar a câmera: volta para a anterior e abre ela
            if atualizador.reverter(RAIZ):
                if acabou_de_atualizar["dependencias_mudaram"]:
                    _instalar_dependencias()
                acabou_de_atualizar = None
                continue
        break


if __name__ == "__main__":
    main()
