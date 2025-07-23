from sign_to_text.detect import detect_sign
from text_to_speech.speak import speak

def run():
    print("Starting Sign-to-Speech System. Press 'q' to quit.")
    text = detect_sign()
    if text:
        print(f"Detected Sign: {text}")
        speak(text)

if __name__ == "__main__":
    run()
