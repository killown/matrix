Matrix Voice Interface (STT/LLM/TTS)

Pipeline assíncrono para interação por voz em tempo real utilizando modelos de inferência local. O foco do projeto é a redução de latência no ciclo completo de voz (Voice-to-Voice) e a estabilidade de captura em ambientes com ruído.
Implementação Técnica:

    Calibração Dinâmica de Ruído: Algoritmo para medição de ruído ambiente (RMS/Peak) e ajuste automático de threshold, prevenindo falsos gatilhos e loops de áudio.

    Processamento Multithread: Separação das camadas de inferência (Ollama) e síntese (Piper) via queue.Queue, garantindo que a geração de texto não seja bloqueada pela reprodução do áudio.

    Token Streaming & Sentence Buffering: Lógica de processamento de tokens por sentença (., !, ?) para início imediato do TTS assim que a primeira frase é concluída, otimizando o tempo de resposta (TTFT).

    Stack Local-First: Integração de OpenAI Whisper (STT), Ollama (LLM) e Piper-TTS (ONNX) para operação offline e privacidade de dados.

Dependências:

    Python 3.13+

    openai-whisper

    sounddevice / numpy

    ollama-python

    piper-tts (CLI)
