#include <Arduino.h>
#include <ESP_I2S.h>
#include <esp_timer.h>
#include <freertos/ringbuf.h>
#include "modelo.h"

static const unsigned AMOSTRAS = 8000;  // 0,5 segundo a 16 kHz.
static const int PINO_BCLK = 26;
static const int PINO_WS = 25;
static const int PINO_SD = 33;
static const int PINO_LED = 2;
static const float RMS_BRILHO_MAXIMO = 0.15f;  // Ganho visual do LED; não é limiar de detecção.
struct Audio {
  int32_t amostras[AMOSTRAS];
  uint32_t sequencia;
  int64_t fim_us;
  uint32_t captura_us;
};
struct Features {
  float rms;
  float zcr;
  uint32_t sequencia;
  int64_t fim_us;
  uint32_t captura_us;
  uint32_t features_us;
};
static I2SClass microfone;
static Audio captura;
static RingbufHandle_t buffer_audio;
static QueueHandle_t fila_features;

// 1. Captura: o buffer circular copia a janela e cuida da sua memória.
static void tarefa_captura(void *) {
  for (;;) {
    int64_t inicio = esp_timer_get_time();
    size_t bytes = microfone.readBytes(reinterpret_cast<char *>(captura.amostras),
                                      sizeof(captura.amostras));
    captura.fim_us = esp_timer_get_time();
    captura.captura_us = captura.fim_us - inicio;
    ++captura.sequencia;  // Uma janela perdida interrompe a contagem de evidências.
    if (bytes == sizeof(captura.amostras) && captura.captura_us <= 750000) {
      xRingbufferSend(buffer_audio, &captura, sizeof(captura), 0);
    } else {
      vTaskDelay(pdMS_TO_TICKS(1));
    }
  }
}

// 2. Features: remover a média evita que o deslocamento DC altere RMS e ZCR.
static void tarefa_features(void *) {
  for (;;) {
    size_t tamanho;
    Audio *audio = static_cast<Audio *>(xRingbufferReceive(buffer_audio, &tamanho, portMAX_DELAY));
    int64_t inicio = esp_timer_get_time();
    double soma = 0;
    bool tem_sinal = false;
    for (unsigned i = 0; i < AMOSTRAS; ++i) {
      int32_t amostra = audio->amostras[i] >> 8;  // INMP441: 24 bits em palavra de 32 bits.
      soma += amostra;
      if (amostra != 0) tem_sinal = true;
    }
    float media = soma / AMOSTRAS;
    double quadrados = 0;
    unsigned cruzamentos = 0;
    float anterior = 0;
    for (unsigned i = 0; i < AMOSTRAS; ++i) {
      float atual = ((audio->amostras[i] >> 8) - media) / 8388608.0f;
      quadrados += atual * atual;
      if (i > 0 && (atual >= 0) != (anterior >= 0)) ++cruzamentos;
      anterior = atual;
    }
    Features janela;
    janela.rms = sqrt(quadrados / AMOSTRAS);
    janela.zcr = float(cruzamentos) / (AMOSTRAS - 1);
    janela.sequencia = audio->sequencia;
    janela.fim_us = audio->fim_us;
    janela.captura_us = audio->captura_us;
    janela.features_us = esp_timer_get_time() - inicio;
    vRingbufferReturnItem(buffer_audio, audio);
    // Zeros digitais podem indicar fio solto: nunca contar como impressora parada.
    if (tem_sinal) {
      xQueueSend(fila_features, &janela, 0);
    } else {
      Serial.println("# microfone: janela com zeros digitais");
    }
  }
}

// 3. Detecção: indicar impressão pela maioria das últimas dez janelas válidas.
static void tarefa_deteccao(void *) {
  static const unsigned JANELAS_VOTO = 10;
  bool votos[JANELAS_VOTO] = {};
  unsigned posicao = 0;
  unsigned quantidade = 0;
  unsigned votos_imprimindo = 0;
  enum class Estado { desconhecido, imprimindo, parada };
  Estado estado = Estado::desconhecido;
  uint32_t sequencia_anterior = 0;
  int64_t tempo_anterior = 0;
  for (;;) {
    Features janela;
    if (xQueueReceive(fila_features, &janela, pdMS_TO_TICKS(800)) != pdTRUE) {
      posicao = 0;
      quantidade = 0;
      votos_imprimindo = 0;
      if (modelo_treinado) {
        digitalWrite(PINO_LED, LOW);
        if (estado == Estado::imprimindo) {
          estado = Estado::parada;
          Serial.println("# ESTADO: parada");
        }
      } else {
        analogWrite(PINO_LED, 0);
      }
      continue;
    }
    int64_t inicio = esp_timer_get_time();
    bool antiga = inicio - janela.fim_us > 750000;
    if (janela.sequencia != sequencia_anterior + 1 ||
        janela.fim_us - tempo_anterior > 750000 || antiga) {
      posicao = 0;
      quantidade = 0;
      votos_imprimindo = 0;
      if (modelo_treinado) {
        digitalWrite(PINO_LED, LOW);
        if (estado == Estado::imprimindo) {
          estado = Estado::parada;
          Serial.println("# ESTADO: parada");
        }
      }
    }
    sequencia_anterior = janela.sequencia;
    tempo_anterior = janela.fim_us;
    uint32_t inferencia_us = 0;
    if (!modelo_treinado && !antiga) {
      int brilho = constrain(int(janela.rms * 255.0f / RMS_BRILHO_MAXIMO), 0, 255);
      analogWrite(PINO_LED, brilho);
    }
    if (modelo_treinado && !antiga) {
      int64_t inicio_inferencia = esp_timer_get_time();
      float logit = bias_modelo + pesos_modelo[0] * janela.rms + pesos_modelo[1] * janela.zcr;
      float probabilidade = 1.0f / (1.0f + expf(-logit));
      inferencia_us = esp_timer_get_time() - inicio_inferencia;
      // Leituras incertas também contam como ausência de impressão.
      if (quantidade == JANELAS_VOTO) {
        votos_imprimindo -= votos[posicao];
      } else {
        ++quantidade;
      }
      votos[posicao] = probabilidade >= 0.7f;
      votos_imprimindo += votos[posicao];
      posicao = (posicao + 1) % JANELAS_VOTO;
      if (quantidade == JANELAS_VOTO) {
        Estado novo_estado = votos_imprimindo > JANELAS_VOTO / 2
                                 ? Estado::imprimindo : Estado::parada;
        if (estado != novo_estado) {
          estado = novo_estado;
          digitalWrite(PINO_LED, estado == Estado::imprimindo ? HIGH : LOW);
          Serial.println(estado == Estado::imprimindo
                             ? "# ESTADO: imprimindo" : "# ESTADO: parada");
        }
      }
    }
    Serial.printf("DADOS,%.8f,%.8f,%lu,%lu,%lu\n", janela.rms, janela.zcr,
                  (unsigned long)janela.captura_us, (unsigned long)janela.features_us,
                  (unsigned long)inferencia_us);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(PINO_LED, OUTPUT);
  digitalWrite(PINO_LED, LOW);
  // NOSPLIT limita cada item à metade da capacidade menos seu cabeçalho.
  buffer_audio = xRingbufferCreate(2 * sizeof(Audio) + 64, RINGBUF_TYPE_NOSPLIT);
  fila_features = xQueueCreate(2, sizeof(Features));
  microfone.setPins(PINO_BCLK, PINO_WS, -1, PINO_SD);
  if (!buffer_audio || !fila_features ||
      !microfone.begin(I2S_MODE_STD, 16000, I2S_DATA_BIT_WIDTH_32BIT,
                       I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("# erro na inicialização das filas ou I2S");
    Serial.flush();
    abort();
  }
  Serial.println("# RMS e ZCR; janelas de 0,5 s; captura_us inclui espera I2S");
  if (modelo_treinado) {
    Serial.println("# modo detecção: LED ligado = pelo menos 6 das últimas 10 janelas imprimindo; demais = desligado");
  } else {
    Serial.println("# modelo não treinado: coleta sem alertas");
  }
  if (xTaskCreate(tarefa_captura, "captura", 4096, nullptr, 3, nullptr) != pdPASS ||
      xTaskCreate(tarefa_features, "features", 4096, nullptr, 2, nullptr) != pdPASS ||
      xTaskCreate(tarefa_deteccao, "deteccao", 4096, nullptr, 1, nullptr) != pdPASS) {
    Serial.println("# erro na criação das tarefas");
    Serial.flush();
    abort();
  }
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}
