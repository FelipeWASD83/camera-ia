@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Instalando o Camera IA (uns 3 GB com o PyTorch para placa NVIDIA)...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv || goto erro
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto erro
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto erro
".venv\Scripts\python.exe" baixar_modelos.py || goto erro
echo.
echo Pronto. Abra pelo atalho do Menu Iniciar ou com: .venv\Scripts\pythonw.exe iniciar.pyw
pause
exit /b 0
:erro
echo.
echo A instalacao parou com erro. Veja a mensagem acima.
pause
exit /b 1
