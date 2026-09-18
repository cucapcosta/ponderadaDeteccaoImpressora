# Detector de parada da impressora

## 1. O que foi feito

Sistema com ESP32 e microfone INMP441 para reconhecer o som de uma impressora 3D funcionando e indicar a impressão pelo LED no GPIO2.

O firmware usa três tarefas FreeRTOS: captura de áudio (prioridade 3), cálculo de RMS e cruzamentos por zero (ZCR, prioridade 2), e detecção com regressão logística (prioridade 1). Essas prioridades foram preservadas nas mudanças do LED; a captura tem a maior prioridade.

No modo de detecção, cada janela tem 0,5 segundo. Uma leitura conta como impressão quando sua probabilidade é pelo menos 0,7. O LED segue a votação das últimas dez janelas válidas (cerca de 5 segundos):

| Resultado | LED no GPIO2 |
| --- | --- |
| 6 ou mais leituras indicam impressão | Ligado |
| 5 ou menos leituras indicam impressão | Desligado |

Leituras incertas contam como ausência de impressão. Essa regra substituiu a exigência de leituras consecutivas e a manutenção do último estado, que podia deixar o LED ligado indefinidamente. Cada mudança de estado gera uma mensagem na serial. O LED começa apagado e só pode acender após completar dez janelas. Sem dados por 800 ms, com janelas antigas ou com perda de sequência, ele apaga e reinicia a votação.

No modo sem modelo treinado, o brilho acompanha o RMS por PWM, com saturação em RMS 0,15. Esse modo reaproveita o cálculo existente, sem criar outra tarefa. Com modelo treinado, o LED indica o resultado da detecção.

Um script Python permite coletar dados, treinar e testar o modelo. A coleta salva as características em CSV, não o áudio completo. O treino gera o arquivo ONNX e os pesos usados pelo ESP.

### Como a detecção funciona

A ideia é que o ESP32 consiga captar, processar e identificar o som da impressora sozinho, sem depender do computador durante o uso. O INMP441 envia áudio digital por I²S. O firmware trabalha com blocos de 0,5 segundo e calcula o RMS, que representa a intensidade do sinal, e o ZCR, que mede quantas vezes o sinal cruza o zero. Nas coletas, são salvos esses valores e o rótulo `imprimindo` ou `parada`, em vez do áudio bruto.

Com esses dados, treinamos uma regressão logística supervisionada no computador. Como o modelo é uma conta pequena, os pesos aprendidos são exportados para C++ e o ESP32 calcula a previsão diretamente, sem precisar de uma biblioteca maior para executar essa fórmula. O modelo também é exportado para ONNX e executado no computador para validar a exportação; o ESP32 usa os mesmos parâmetros aprendidos, mas não abre o arquivo ONNX.

O trabalho é separado em três tarefas FreeRTOS: captura, cálculo das características e detecção. A captura recebe a maior prioridade para não perder amostras do microfone. Por fim, o LED usa a maioria das últimas dez leituras para não mudar de estado por causa de uma previsão isolada.

### Resultado e limitação observada

Foram feitas coletas reais, retreinamento e testes exploratórios com a impressora. No uso observado, movimentos maiores, como ir de um canto ao outro ou mudar entre partes da peça, tendem a ser reconhecidos como **imprimindo**. Movimentos menores, mais silenciosos ou concentrados na mesma região podem não ser reconhecidos, mesmo durante a impressão. Essa é uma observação prática, sem comprovação da causa.

Portanto, o LED apagado indica ausência de impressão reconhecida pela regra de votação; não comprova que a impressora parou fisicamente, terminou a peça ou apresentou erro. A confiabilidade ainda é limitada.

O último teste em duas sessões reservadas teve **70,7% de acurácia por janela**: 97/121 acertos para parada e 74/121 para impressão, conforme `modelo/metricas.json`. Essa métrica avalia classificações individuais, não a votação temporal nem a confiabilidade do LED em uso contínuo.

## 2. Como rodar

Conecte o INMP441 ao ESP32 DevKit/WROOM:

| INMP441 | ESP32 |
| --- | --- |
| VDD | 3V3 |
| GND e L/R | GND |
| SCK / BCLK | GPIO26 |
| WS | GPIO25 |
| SD | GPIO33 |

O LED usa GPIO2. Se precisar de LED externo, conecte-o com resistor de 330 Ω em série até GND.

Na Arduino IDE, instale **esp32 by Espressif Systems 3.3.11**, selecione **ESP32 Dev Module** e grave `firmware/detector/detector.ino`. O `modelo.h` incluído já está treinado, então o firmware inicia no modo de detecção. A serial continua emitindo características para novas coletas.

Na raiz do projeto, prepare o Python e abra o menu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/modelo.py
```

Para coletar novos dados e retreinar, use as opções nesta ordem:

1. **Coletar:** informe a porta serial e a classe. Faça pelo menos duas coletas de 60 segundos com a impressora funcionando e duas parada, em sessões distintas. Inclua parada com ventoinhas ligadas. Feche o monitor serial durante a coleta.
2. **Treinar:** gera `modelo/modelo.onnx`, `modelo/metricas.json` e `firmware/detector/modelo.h`.
3. **Testar:** avalia as sessões reservadas para teste e mede o tempo de inferência no PC.

Execute `python scripts/modelo.py` novamente para escolher outra opção. **Após treinar, compile e grave o sketch novamente no ESP32.** Abra o monitor serial em **115200 baud** para acompanhar os dados e o alerta.

### Uso de IA

Fora utilizado agentes de IA, junto com Wisprflow e equivalentes, para auxílio de documentação e assistências na parte _pesada_ de código, além de ajudar na ideia da captura de áudio da forma que foi executada