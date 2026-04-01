import subprocess
import ollama
import re
import threading
import queue
import whisper
import sounddevice as sd
import numpy as np
import time

# --- CONFIG ---
MODELO_OLLAMA = "samuser3/granite3.2-gemma3:4b"
VOZ_PATH = "pt_BR-faber-medium.onnx"

audio_queue = queue.Queue()
historico = [
    {
        "role": "system",
        "content": "Você é Matrix. Terminal técnico. Respostas curtas e objetivas.",
    }
]
threshold_dinamico = 0.01

print("--- [SISTEMA] Inicializando Matrix STT ---")
stt_model = whisper.load_model("base")


def calibrar_ruido():
    """Mede o ruído do ambiente para evitar loops infinitos."""
    global threshold_dinamico
    print("[CALIBRANDO SILÊNCIO...] Aguarde 1.5s...")
    fs = 16000
    audio = sd.rec(int(1.5 * fs), samplerate=fs, channels=1, blocking=True)
    ruido_base = np.max(np.abs(audio))
    # Define o corte acima do ruído de fundo
    threshold_dinamico = ruido_base * 1.5 + 0.002
    print(f"[OK] Sensibilidade ajustada.")


def worker_audio():
    """Thread de TTS (Piper)"""
    while True:
        texto = audio_queue.get()
        if texto is None:
            break
        t_limpo = re.sub(r"[*#_>`-]", "", texto).strip()
        if t_limpo:
            command = (
                f'echo "{t_limpo}" | piper-tts --model {VOZ_PATH} --output_raw 2>/dev/null | '
                f"aplay -r 22050 -f S16_LE -t raw -c 1 2>/dev/null"
            )
            subprocess.run(command, shell=True)
        audio_queue.task_done()


def ouvir_dinamico():
    """Captura áudio baseada em volume."""
    fs = 16000
    chunk_size = 1024
    audio_data = []

    with sd.InputStream(samplerate=fs, channels=1, dtype="float32") as stream:
        silence_counter = 0
        recording = False

        while True:
            chunk, _ = stream.read(chunk_size)
            vol = np.max(np.abs(chunk))

            if vol > threshold_dinamico:
                recording = True
                silence_counter = 0
                audio_data.append(chunk)
            elif recording:
                audio_data.append(chunk)
                silence_counter += 1

            # Para ao detectar silêncio (~0.8s)
            if recording and silence_counter > 12:
                break

            # Timeout de espera
            if not recording and silence_counter > 100:
                return ""

    if not audio_data:
        return ""

    full_audio = np.concatenate(audio_data).flatten()
    result = stt_model.transcribe(full_audio, language="pt", fp16=False, best_of=1)
    return result["text"].strip()


def chat():
    global historico
    threading.Thread(target=worker_audio, daemon=True).start()

    calibrar_ruido()
    print("\n[ON] MATRIX: ESCUTA DIRETA ATIVA")

    while True:
        try:
            if audio_queue.empty():
                comando = ouvir_dinamico()

                if not comando or len(comando.split()) < 2:
                    continue

                print(f"\n>> VOCÊ: {comando}")
                print(f">> MATRIX: ", end="", flush=True)

                historico.append({"role": "user", "content": comando})
                full_response = ""
                buffer_frase = ""

                for chunk in ollama.chat(
                    model=MODELO_OLLAMA, messages=historico, stream=True
                ):
                    token = chunk["message"]["content"]
                    print(token, end="", flush=True)
                    full_response += token
                    buffer_frase += token

                    if any(p in token for p in [".", "!", "?", "\n", ":"]):
                        if len(buffer_frase.strip()) > 1:
                            audio_queue.put(buffer_frase.strip())
                        buffer_frase = ""

                historico.append({"role": "assistant", "content": full_response})
                audio_queue.join()
                print("\n")
            else:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[OFF] Encerrado.")
            break


if __name__ == "__main__":
    chat()
