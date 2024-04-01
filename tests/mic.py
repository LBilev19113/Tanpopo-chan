import sre_compile
import pygame
import pygame._sdl2.audio as sdl2_audio
from time import sleepfrom
from gtts import gTTS

def get_devices(capture_devices: bool = False) -> tuple[str, ...]:
    init_by_me = not pygame.mixer.get_init()
    if init_by_me:
        pygame.mixer.init()
    devices = tuple(sdl2_audio.get_audio_device_names(capture_devices))
    if init_by_me:
        pygame.mixer.quit()
    return devices[2]

def play(file_path: str, device: str):
    if device is None:
        devices = get_devices()
        if not devices:
            raise RuntimeError("No device!")
        device = devices[0]
    print("Play: {}\r\nDevice: {}".format(file_path, device))
    pygame.mixer.init(devicename=device)
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    try:
        while True:
            sleep(0.1)
    except KeyboardInterrupt:
        pass
    pygame.mixer.quit()

play("output.mp3", "CABLE Input (VB-Audio Virtual Cable)")

def convert_audio(text):
    tts = gTTS(text, lang='en', slow=False)
    tts.save("output.mp3")


def recognize_speech():
    recognizer = sre_compile.Recognizer()


    with sr.Microphone() as source:
        print("Say something:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio)
        print("You said:", query)
        return query.lower()
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return "Sorry, I couldn't understand that."
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""