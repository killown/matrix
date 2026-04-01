## Matrix Voice Interface (STT/LLM/TTS)

Pipeline assíncrono para interação por voz em tempo real via modelos locais. Foco em baixa latência e estabilidade em ambientes ruidosos.

### Implementação Técnica
* **Calibração de Ruído:** Algoritmo para medição de ruído ambiente (RMS/Peak) e ajuste automático de threshold.
* **Multithreading:** Separação das camadas de inferência (Ollama) e síntese (Piper) via `queue.Queue`.
* **Token Streaming:** Processamento por sentença (`.`, `!`, `?`) para início imediato do TTS (otimização de TTFT).
* **Local-First:** Integração de Whisper (STT), Ollama (LLM) e Piper (ONNX) para operação offline.

### Dependências
* Python 3.13+
* `openai-whisper`
* `sounddevice` / `numpy`
* `ollama-python`
* `piper-tts` (CLI)

### *  ❯❯❯ Baixa o modelo de voz (ONNX)
      wget https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx
   
### *  ❯❯❯   Baixa as configurações da voz (JSON):
      wget https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json
