# Câmera IA

Abre a webcam numa janela e uma IA, rodando na placa de vídeo, mostra em tempo real o que está vendo:
uma caixa em volta de cada coisa, com o nome e a porcentagem de certeza em cima. Nas pessoas, cada rosto
ganha outra caixa com a expressão (Feliz, Triste, Raiva, Surpreso, Medo, Nojo, Desprezo ou Neutro) e a certeza.

Tudo roda no PC, nada vai para a internet (só a procura por atualização, no GitHub).

## Abrir

Pelo atalho **Câmera IA** no Menu Iniciar. Se a câmera estiver com a tampa de privacidade fechada, a imagem
fica preta e a IA não acha nada.

| Tecla | Faz |
|---|---|
| F | Tela cheia (aperta de novo para sair) |
| U | Procura versão nova no GitHub |
| A | Instala a versão nova (só aparece quando tem uma) |
| Q ou Esc | Fecha (o X da janela também) |

No canto de baixo aparecem o fps do vídeo e o da IA, e quanto tempo a IA leva por quadro.

## Instalar do zero

Precisa de Python 3.13 e placa NVIDIA. Rode `instalar.bat`: ele cria o `.venv`, instala o PyTorch para
CUDA e as outras bibliotecas (uns 3 GB) e baixa os três modelos para `modelos/` (uns 70 MB, conferidos
pelo SHA-256).

## Ajustes

Ficam em `dados/config.json` (criado na primeira vez que o app abre). Fecha o app, edita, abre de novo.

| Chave | Padrão | Para que serve |
|---|---|---|
| `camera` | `0` | Qual câmera usar, se tiver mais de uma (0, 1, 2...) |
| `largura`, `altura` | `1920`, `1080` | Resolução pedida para a câmera |
| `modelo` | `yolo26l.pt` | Modelo do YOLO. `yolo26x.pt` acerta um pouco mais e pesa mais; tem que estar em `modelos/` |
| `confianca_minima` | `0.40` | Abaixo disso o objeto não ganha caixa. Mais alto = menos caixa errada, mais coisa perdida |
| `confianca_rosto` | `0.75` | O mesmo para achar rosto |
| `mostrar_fps` | `true` | Mostra o fps no canto |

## Os modelos

- **YOLO26 large** (Ultralytics): acha as 80 coisas do conjunto COCO (pessoa, celular, garrafa, cadeira,
  cachorro, notebook...). Coisa fora dessa lista ele não reconhece, ou chama pelo nome mais parecido.
- **YuNet** (OpenCV): acha os rostos.
- **HSEmotion EfficientNet-B0** (`enet_b0_8_best_vgaf`): lê a expressão. É o que mais acertou num teste com
  fotos de expressão conhecida, mas expressão é coisa difícil: cara séria às vezes sai "Desprezo" ou "Raiva",
  e surpresa às vezes sai "Medo".

Licenças: o YOLO é AGPL-3.0, o YuNet MIT e o HSEmotion Apache-2.0.
