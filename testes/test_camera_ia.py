"""Testes do Câmera IA. Rodar da pasta do projeto: .venv\\Scripts\\python.exe -m unittest discover -s testes"""
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import atualizador, config, desenho, rotulos  # noqa: E402

ASSETS = RAIZ / ".venv" / "Lib" / "site-packages" / "ultralytics" / "assets"


class TestIA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.analise import Analisador
        cls.ia = Analisador(dict(config.PADRAO))

    def test_todas_as_classes_tem_nome_em_portugues(self):
        nomes = self.ia.yolo.names.values()
        faltando = [n for n in nomes if n not in rotulos.COCO]
        self.assertEqual(faltando, [])

    def test_onibus_e_pessoas(self):
        f = cv2.imread(str(ASSETS / "bus.jpg"))
        r = self.ia.analisar(f)
        nomes = [o.nome for o in r.objetos]
        self.assertIn("Ônibus", nomes)
        self.assertGreaterEqual(nomes.count("Pessoa"), 3)
        self.assertTrue(all(0 < o.confianca <= 1 for o in r.objetos))

    def test_rostos_com_expressao(self):
        f = cv2.imread(str(ASSETS / "zidane.jpg"))
        self.ia._anteriores = []
        r = self.ia.analisar(f)
        self.assertEqual(len(r.rostos), 2)
        for rosto in r.rostos:
            self.assertIn(rosto.emocao, rotulos.EMOCOES)
            self.assertAlmostEqual(float(rosto.probs.sum()), 1.0, places=5)
        # Zidane está gritando: a expressão mais provável dele é raiva
        zidane = min(r.rostos, key=lambda x: x.caixa[0])
        self.assertEqual(zidane.emocao, "Raiva")

    def test_rosto_grande_perto_da_camera(self):
        # rosto ocupando quase a imagem toda: o YuNet em escala cheia não achava
        f = cv2.imread(str(ASSETS / "zidane.jpg"))[230:460, 520:720]
        f = cv2.resize(f, (1080, 1240))
        self.assertGreaterEqual(len(self.ia._caixas_rostos(f)), 1)

    def test_desenho_nao_mexe_no_quadro_original(self):
        f = cv2.imread(str(ASSETS / "zidane.jpg"))
        copia = f.copy()
        img = desenho.desenhar(f, self.ia.analisar(f))
        self.assertEqual(img.shape, f.shape)
        self.assertTrue(np.array_equal(f, copia))
        self.assertFalse(np.array_equal(img, f))


class TestAtualizador(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raiz = self.tmp / "app-instalado"
        (self.raiz / "app").mkdir(parents=True)
        (self.raiz / "app" / "principal.py").write_text("velho", encoding="utf-8")
        (self.raiz / "iniciar.pyw").write_text("velho", encoding="utf-8")
        (self.raiz / "versao.json").write_text('{"versao": "1.0.0"}', encoding="utf-8")
        (self.raiz / "dados").mkdir()
        (self.raiz / "dados" / "config.json").write_text("{}", encoding="utf-8")
        pasta = self.tmp / "atualizacao"
        pasta.mkdir()
        self.patches = [mock.patch.object(atualizador, "PASTA", pasta),
                        mock.patch.object(atualizador, "_PRONTA", pasta / "pronta.json")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_comparar_versoes(self):
        self.assertTrue(atualizador.mais_nova("1.0.10", "1.0.9"))
        self.assertTrue(atualizador.mais_nova("v2.0.0", "1.9.9"))
        self.assertFalse(atualizador.mais_nova("1.0.0", "1.0.0"))

    def _versao_nova(self):
        novo = atualizador.PASTA / "novo"
        (novo / "app").mkdir(parents=True)
        (novo / "app" / "principal.py").write_text("novo", encoding="utf-8")
        (novo / "iniciar.pyw").write_text("novo", encoding="utf-8")
        (novo / "versao.json").write_text('{"versao": "1.1.0"}', encoding="utf-8")
        atualizador._PRONTA.write_text(json.dumps({"versao": "1.1.0", "pasta": str(novo)}), encoding="utf-8")

    def test_aplicar_e_reverter_sem_tocar_nos_dados(self):
        self._versao_nova()
        info = atualizador.aplicar_pendente(self.raiz)
        self.assertEqual(info["versao"], "1.1.0")
        self.assertEqual((self.raiz / "app" / "principal.py").read_text(encoding="utf-8"), "novo")
        self.assertTrue((self.raiz / "dados" / "config.json").exists())
        self.assertTrue(atualizador.reverter(self.raiz))
        self.assertEqual((self.raiz / "app" / "principal.py").read_text(encoding="utf-8"), "velho")
        self.assertEqual((self.raiz / "iniciar.pyw").read_text(encoding="utf-8"), "velho")
        self.assertTrue((self.raiz / "dados" / "config.json").exists())

    def test_copia_de_desenvolvimento_nao_e_trocada(self):
        (self.raiz / ".git").mkdir()
        self._versao_nova()
        self.assertIsNone(atualizador.aplicar_pendente(self.raiz))
        self.assertEqual((self.raiz / "app" / "principal.py").read_text(encoding="utf-8"), "velho")

    def test_pacote_com_caminho_para_fora_e_recusado(self):
        zip_path = atualizador.PASTA / "pacote.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("../fora.txt", "x")
        info = {"ultima": "9.9.9", "nova": True, "pacote": {"url": "x", "digest": None}}
        with mock.patch.object(atualizador, "copia_de_desenvolvimento", return_value=False), \
                mock.patch.object(atualizador, "checar", return_value=info), \
                mock.patch.object(atualizador, "_baixar", lambda url, destino: None):
            with self.assertRaises(atualizador.ErroAtualizacao):
                atualizador.baixar_e_preparar()
        self.assertFalse((atualizador.PASTA / "fora.txt").exists())


if __name__ == "__main__":
    unittest.main()
