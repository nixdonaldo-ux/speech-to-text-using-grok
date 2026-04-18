import os
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from pathlib import Path
import tempfile
import time

def grok_stt(audio_path: str) -> dict:
    """Transcribe any audio file using Grok STT (English by default)."""
    url = "https://api.x.ai/v1/stt"
    headers = {"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"}
    
    # Auto-detect MIME type (works for mp3, wav, m4a, etc.)
    ext = Path(audio_path).suffix.lower()
    mime = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
    }.get(ext, "audio/wav")
    
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f, mime)}
        response = requests.post(url, headers=headers, files=files)
    
    response.raise_for_status()
    result = response.json()
    
    print("\n✅ Transcription:")
    print(result["text"])
    print(f"\nDuration: {result.get('duration', 'N/A')} seconds")
    
    # Optional: word-level timestamps & confidence
    if "words" in result:
        print("\nWord-level details available in result['words']")
    
    return result


def record_from_mic(duration: int = None, fs: int = 16000) -> str:
    """Record from microphone and save as temporary WAV."""
    print("🎙️  Recording from microphone... (press Ctrl+C to stop early)")
    
    if duration is None:
        # Record until you press Ctrl+C
        print("   Recording until you stop (Ctrl+C)...")
        try:
            recording = []
            while True:
                chunk = sd.rec(int(fs * 0.5), samplerate=fs, channels=1, dtype='int16')
                sd.wait()
                recording.append(chunk)
        except KeyboardInterrupt:
            print("\n⏹️  Recording stopped.")
            recording = np.concatenate(recording, axis=0)
    else:
        # Fixed duration
        print(f"   Recording for {duration} seconds...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav.write(tmp.name, fs, recording)
        temp_path = tmp.name
    
    print(f"   Saved to temporary file: {temp_path}")
    return temp_path


def main():
    print("🚀 Grok Speech-to-Text Agent (English)")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Transcribe existing audio file")
        print("2. Record from microphone & transcribe")
        print("3. Quit")
        
        choice = input("\nEnter 1, 2, or 3: ").strip()
        
        if choice == "1":
            path = input("Enter audio file path (mp3/wav/etc.): ").strip()
            if Path(path).exists():
                grok_stt(path)
            else:
                print("❌ File not found!")
        
        elif choice == "2":
            try:
                temp_file = record_from_mic()  # or record_from_mic(duration=10) for fixed time
                grok_stt(temp_file)
                # Clean up temp file
                Path(temp_file).unlink(missing_ok=True)
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    # Make sure API key is set
    if not os.getenv("XAI_API_KEY"):
        print("❌ Please set XAI_API_KEY environment variable!")
    else:
        main()
