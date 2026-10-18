# CLAUDE.md: camera-ia

App de janela em Python: webcam + IA em tempo real. YOLO26 acha os objetos (caixa, nome em português,
porcentagem), YuNet acha os rostos e o HSEmotion lê a expressão de cada um. Roda só no PC do Felipe
(RTX 3060, webcam Logitech C920). Versão em `versao.json` e no topo do `CHANGELOG.md` (andam juntos).

## Como roda

```
atalho do Menu Iniciar ─► .venv\Scripts\pythonw.exe iniciar.pyw   (aplica atualização pendente, volta se falhar)
                              └─► pythonw -m app.principal        (janela; saída em dados/log.txt)
```

- `app/camera.py`: thread que lê a webcam (MSMF, MJPG) e guarda o quadro mais novo.
- `app/analise.py`: `Analisador.analisar(quadro)` → `Resultado(objetos, rostos, ms)`.
- `app/principal.py`: thread da IA (pega sempre o quadro mais novo) + laço da janela (desenha cada quadro da
  câmera com o último resultado). O vídeo nunca espera a IA.
- `app/desenho.py`: caixas pelo OpenCV, texto pelo Pillow (Segoe UI) em etiquetas pequenas com cache.
- `app/atualizador.py`: cópia adaptada do `clave/servidor/atualizador.py` (releases do GitHub, SHA-256,
  pasta `anterior` para voltar). Não toca em `dados/`, `modelos/` nem `.venv/`. Cópia com `.git` (esta
  pasta) não se atualiza pelo app: é git pull.
- `baixar_modelos.py`: baixa os 3 modelos com SHA-256; o app chama sozinho se faltar algum.
- Testes: `.venv\Scripts\python.exe -m unittest discover -s testes` (usam as fotos que vêm no Ultralytics).
- Medir: `.venv\Scripts\python.exe -m app.principal --sair-em 15 --salvar-quadro x.png` imprime a mediana do fps.

## Decisões que não se desfazem sem motivo

- **Versões fixas no `requirements.txt` por causa do Smart App Control.** torch 2.8.0+cu128 e scipy 1.15.3
  são as que passam no ComfyUI; OpenCV 4.12 (o 5.0 é novo demais para ter reputação); onnxruntime-gpu é
  assinado pela Microsoft. Testar o import antes de subir qualquer versão.
- **`OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0`** (em `camera.py`, antes do import do cv2): sem isso a C920
  levava 56 s para abrir pelo MSMF. Pelo DSHOW ela abre mas fica em 5 fps.
- **Emoção pelo ONNX Runtime, não pelo `cv2.dnn`**: mesma saída (diferença 2e-6), metade do tempo na CPU e roda
  na GPU. O `import torch` vem antes do `onnxruntime` de propósito (é ele que carrega as DLLs do CUDA/cuDNN).
- **`cv2.setNumThreads(8)` depois do Ultralytics**: ele põe o OpenCV em 1 thread ao ser importado.
- **Modelo de emoção `enet_b0_8_best_vgaf`.** Comparado em 18/10/2026 com `enet_b2_8`, `enet_b2_7`,
  `enet_b0_8_best_afew` e `enet_b0_8_va_mtl` em fotos de expressão conhecida (Wikimedia): acertou 6/6 sorrisos
  a 100% e as 2 fotos tristes. O `enet_b2_8` dava probabilidades achatadas (sorriso escancarado a 35%).
- **Rostos em escala fixa de 640 px + segunda busca dentro de cada pessoa sem rosto (480 px).** O YuNet na
  escala cheia perde rosto grande (perto da câmera) e em 640 perde rosto longe.
- **Margem de 10% no recorte do rosto** e média com o quadro anterior (IoU > 0,3) para a expressão não piscar.
- **Janela com nome interno `camera-ia`** e título trocado por `SetWindowTextW`: o OpenCV escreve o título em
  ANSI ("CÃ¢mera IA"). Ícone da janela vem de `icone/icone.ico` (gerado por
  `_organizacao\atalhos\gerar_icones.py camera-ia`, que copia para cá).

## Desempenho medido (18/10/2026, 1920x1080)

Com a GPU livre: YOLO26l fp16 16 ms, rostos 8 ms, emoção 5 ms para 2 rostos, desenho 3 ms. App real:
**vídeo 30 fps e IA 30 fps** (25 ms por quadro), mediana de 12 s. O teto é a câmera (a C920 dá 30 fps em
1080p), não o PC; a IA analisa todo quadro. Com o Rocket League aberto a GPU fica em 97% e o YOLO sobe para
~100 a 140 ms: é contenção, não defeito.
