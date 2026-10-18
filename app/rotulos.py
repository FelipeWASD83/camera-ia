"""Nomes em português das 80 classes do COCO (as que o YOLO conhece) e das 8 expressões do AffectNet."""

COCO = {
    "person": "Pessoa", "bicycle": "Bicicleta", "car": "Carro", "motorcycle": "Moto", "airplane": "Avião",
    "bus": "Ônibus", "train": "Trem", "truck": "Caminhão", "boat": "Barco", "traffic light": "Semáforo",
    "fire hydrant": "Hidrante", "stop sign": "Placa de pare", "parking meter": "Parquímetro", "bench": "Banco",
    "bird": "Pássaro", "cat": "Gato", "dog": "Cachorro", "horse": "Cavalo", "sheep": "Ovelha", "cow": "Vaca",
    "elephant": "Elefante", "bear": "Urso", "zebra": "Zebra", "giraffe": "Girafa", "backpack": "Mochila",
    "umbrella": "Guarda-chuva", "handbag": "Bolsa", "tie": "Gravata", "suitcase": "Mala", "frisbee": "Frisbee",
    "skis": "Esqui", "snowboard": "Snowboard", "sports ball": "Bola", "kite": "Pipa", "baseball bat": "Taco de beisebol",
    "baseball glove": "Luva de beisebol", "skateboard": "Skate", "surfboard": "Prancha de surfe",
    "tennis racket": "Raquete", "bottle": "Garrafa", "wine glass": "Taça", "cup": "Copo", "fork": "Garfo",
    "knife": "Faca", "spoon": "Colher", "bowl": "Tigela", "banana": "Banana", "apple": "Maçã",
    "sandwich": "Sanduíche", "orange": "Laranja", "broccoli": "Brócolis", "carrot": "Cenoura",
    "hot dog": "Cachorro-quente", "pizza": "Pizza", "donut": "Rosquinha", "cake": "Bolo", "chair": "Cadeira",
    "couch": "Sofá", "potted plant": "Planta", "bed": "Cama", "dining table": "Mesa", "toilet": "Vaso sanitário",
    "tv": "TV", "laptop": "Notebook", "mouse": "Mouse", "remote": "Controle remoto", "keyboard": "Teclado",
    "cell phone": "Celular", "microwave": "Micro-ondas", "oven": "Forno", "toaster": "Torradeira", "sink": "Pia",
    "refrigerator": "Geladeira", "book": "Livro", "clock": "Relógio", "vase": "Vaso", "scissors": "Tesoura",
    "teddy bear": "Ursinho de pelúcia", "hair drier": "Secador de cabelo", "toothbrush": "Escova de dentes",
}

# Ordem de saída do enet_b0_8_best_vgaf (HSEmotion: AffectNet, ajustado no VGAF).
EMOCOES = ["Raiva", "Desprezo", "Nojo", "Medo", "Feliz", "Neutro", "Triste", "Surpreso"]

# Cor da caixa do rosto por expressão, em BGR.
COR_EMOCAO = {
    "Raiva": (40, 40, 220), "Desprezo": (120, 60, 150), "Nojo": (40, 140, 90), "Medo": (160, 80, 200),
    "Feliz": (40, 200, 255), "Neutro": (190, 190, 190), "Triste": (220, 130, 40), "Surpreso": (60, 220, 160),
}


def nome(classe_en: str) -> str:
    return COCO.get(classe_en, classe_en.capitalize())
