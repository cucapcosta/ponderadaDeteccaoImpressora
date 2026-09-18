"""Coleta, treinamento e teste do detector de impressora."""
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import serial
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import to_onnx

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
SAIDA = RAIZ / "modelo"
CABECALHO = RAIZ / "firmware/detector/modelo.h"
PORTA = "/dev/ttyUSB0"
DURACAO = 60
CLASSES = {"parada": 0, "imprimindo": 1}


def ler_csv(caminho):
    valores = []
    rotulos = []
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        if leitor.fieldnames != ["rms", "zcr", "classe"]:
            raise ValueError(f"{caminho}: esperado rms,zcr,classe")
        for linha in leitor:
            if None in linha or any(v is None for v in linha.values()):
                raise ValueError(f"{caminho}: linha incompleta ou com colunas extras")
            features = [float(linha["rms"]), float(linha["zcr"])]
            if not all(np.isfinite(v) and 0 <= v <= 1 for v in features):
                raise ValueError(f"{caminho}: RMS/ZCR inválidos")
            valores.append(features)
            rotulos.append(CLASSES[linha["classe"]])
    if len(valores) < 2 or len(set(rotulos)) != 1:
        raise ValueError(f"{caminho}: precisa de pelo menos 2 linhas de uma única classe")
    return valores, rotulos


def coletar():
    porta = input(f"Porta serial [{PORTA}]: ").strip() or PORTA
    escolha = input("Classe: 0 parada, 1 imprimindo: ").strip()
    if escolha not in ("0", "1"):
        raise ValueError("escolha 0 ou 1")
    if escolha == "0":
        classe = "parada"
    else:
        classe = "imprimindo"
    DADOS.mkdir(parents=True, exist_ok=True)
    caminho = DADOS / f"{classe}_{time.time_ns()}.csv"
    quantidade = 0
    with caminho.open("x", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["rms", "zcr", "classe"])
        with serial.Serial(porta, 115200, timeout=1) as conexao:
            inicio = ultimo_dado = time.monotonic()
            while time.monotonic() - inicio < DURACAO:
                if time.monotonic() - ultimo_dado >= 10:
                    raise ValueError("10 segundos sem dados válidos; verifique microfone e porta")
                campos = conexao.readline().decode("utf-8", errors="replace").strip().split(",")
                if len(campos) != 6 or campos[0] != "DADOS":
                    continue
                try:
                    valores = [float(v) for v in campos[1:3]]
                    tempos = [int(v) for v in campos[3:]]
                except ValueError:
                    continue
                if not all(np.isfinite(v) and 0 <= v <= 1 for v in valores):
                    continue
                if any(v < 0 for v in tempos):
                    continue
                escritor.writerow([*valores, classe])
                arquivo.flush()
                quantidade += 1
                ultimo_dado = time.monotonic()
    if quantidade < 2:
        raise ValueError("captura sem dados suficientes")
    print(f"{quantidade} janelas gravadas em {caminho}")


def treinar():
    sessoes = {0: [], 1: []}
    gravacoes = {}
    for caminho in sorted(DADOS.glob("*.csv")):
        gravacoes[caminho] = ler_csv(caminho)
        sessoes[gravacoes[caminho][1][0]].append(caminho)
    if any(len(arquivos) < 2 for arquivos in sessoes.values()):
        raise ValueError("colete pelo menos 2 sessões distintas por classe")
    # Reservar arquivos inteiros evita testar em janelas vizinhas às usadas no treino.
    teste = [arquivos[-1] for arquivos in sessoes.values()]
    dados_treino = []
    rotulos_treino = []
    dados_teste = []
    rotulos_teste = []
    for caminho, (valores, rotulos) in gravacoes.items():
        if caminho in teste:
            dados_teste.extend(valores)
            rotulos_teste.extend(rotulos)
        else:
            dados_treino.extend(valores)
            rotulos_treino.extend(rotulos)
    dados_treino = np.asarray(dados_treino, dtype=np.float32)
    dados_teste = np.asarray(dados_teste, dtype=np.float32)
    normalizador = StandardScaler()
    classificador = LogisticRegression(class_weight="balanced")
    modelo = make_pipeline(normalizador, classificador)
    modelo.fit(dados_treino, rotulos_treino)
    onnx = to_onnx(modelo, dados_treino[:1], options={id(classificador): {"zipmap": False}}).SerializeToString()
    sessao = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    entrada = np.concatenate([dados_treino, dados_teste])
    esperado = modelo.predict_proba(entrada)[:, 1]
    previsto = sessao.run(None, {sessao.get_inputs()[0].name: entrada})[1][:, 1]
    if not np.allclose(previsto, esperado, atol=1e-5, rtol=0):
        raise ValueError("exportação ONNX divergiu do modelo")
    # Incorporar a normalização nos pesos deixa a mesma regressão simples no ESP32.
    pesos = (classificador.coef_[0] / normalizador.scale_).astype(np.float32)
    bias = np.float32(classificador.intercept_[0] - np.dot(pesos, normalizador.mean_))
    probabilidades_exportadas = 1 / (1 + np.exp(-(entrada @ pesos + bias)))
    if not np.allclose(probabilidades_exportadas, esperado, atol=1e-5, rtol=0):
        raise ValueError("pesos exportados divergem do modelo")
    cabecalho = ("#pragma once\nstatic const bool modelo_treinado = true;\n"
                f"static const float pesos_modelo[2] = {{{pesos[0]:.9e}f, {pesos[1]:.9e}f}};\n"
                f"static const float bias_modelo = {bias:.9e}f;\n")
    matriz = confusion_matrix(rotulos_teste, modelo.predict(dados_teste), labels=[0, 1])
    metricas = {"arquivos_teste": [p.name for p in teste],
                "matriz_confusao": matriz.tolist(),
                "acuracia": float(np.trace(matriz) / matriz.sum())}
    SAIDA.mkdir(parents=True, exist_ok=True)
    CABECALHO.parent.mkdir(parents=True, exist_ok=True)
    (SAIDA / "modelo.onnx").write_bytes(onnx)
    (SAIDA / "metricas.json").write_text(json.dumps(metricas, indent=2) + "\n")
    CABECALHO.write_text(cabecalho)
    print(json.dumps(metricas, indent=2))


def testar():
    metricas = json.loads((SAIDA / "metricas.json").read_text())
    valores = []
    rotulos = []
    for nome in metricas["arquivos_teste"]:
        x, y = ler_csv(DADOS / nome)
        valores.extend(x)
        rotulos.extend(y)
    entrada = np.asarray(valores, dtype=np.float32)
    sessao = ort.InferenceSession(str(SAIDA / "modelo.onnx"), providers=["CPUExecutionProvider"])
    nome = sessao.get_inputs()[0].name
    previsto = sessao.run(None, {nome: entrada})[0]
    matriz = confusion_matrix(rotulos, previsto, labels=[0, 1])
    amostra = {nome: entrada[:1]}
    sessao.run(None, amostra)  # Aquecimento antes de medir.
    inicio = time.perf_counter()
    for _ in range(200):
        sessao.run(None, amostra)
    media_us = (time.perf_counter() - inicio) * 1e6 / 200
    print(f"Matriz [parada, imprimindo]:\n{matriz}\n"
          f"Acurácia: {np.trace(matriz) / matriz.sum():.1%}\nInferência PC média: {media_us:.2f} us")


if __name__ == "__main__":
    escolha = input("1 Coletar\n2 Treinar\n3 Testar\nEscolha: ").strip()
    try:
        if escolha == "1":
            coletar()
        elif escolha == "2":
            treinar()
        elif escolha == "3":
            testar()
        else:
            print("Escolha 1, 2 ou 3.")
    except (OSError, ValueError, KeyError, serial.SerialException) as erro:
        print(f"Erro: {erro}")
        raise SystemExit(1)
