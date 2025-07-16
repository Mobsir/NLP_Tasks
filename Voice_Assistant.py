import asyncio
import cv2
import time
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import edge_tts
import os
import pygame
import unicodedata
from datetime import datetime



# Folders to store images
FAMILY_FOLDER = "family"
CAPTURE_FOLDER = "captured_images"
for folder in [FAMILY_FOLDER, CAPTURE_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize pygame mixer for playing TTS audio
pygame.mixer.init()

def normalize_text(text):
     """
    Normalize Arabic text by removing diacritics and converting to lowercase.

    This function uses Unicode normalization to strip combining characters (harakat)
    and then returns the clean, lowercased text.

    Args:
        text (str): Arabic text to normalize.

    Returns:
        str: Normalized text without diacritics and in lowercase.
    """
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()


# List of recognized voice commands (normalized)
START_COMMANDS = [normalize_text(cmd) for cmd in ["أَهْلًا مُبْصِر", "مَرْحَبًا مُبْصِر", "السَّلَامُ عَلَيْك"]]
EXPLORE_COMMANDS = [normalize_text(cmd) for cmd in ["اِسْتَكْشِفْ المَكَان", "اِسْتَكْشَاف المَكَان", "اِسْتِكْشَاف"]]
PHOTO_COMMANDS = [normalize_text(cmd) for cmd in ["اِلْتَقِطْ صُورَة", "صَوِّرْ", "أَخَذْ صُورَة"]]
EXIT_COMMANDS = [normalize_text(cmd) for cmd in ["شُكْرًا مُبْصِر", "إِنْهَاء", "خُرُوج"]]

# Text-to-Speech using Edge TTS
async def edge_speak(text):
    """
    Convert Arabic text to speech using Edge TTS and play it.

    This function creates an MP3 file using Edge TTS with an Arabic voice,
    plays it using pygame, and then deletes the file after playback.

    Args:
        text (str): Arabic text to speak.
    """
    voice = "ar-EG-SalmaNeural"
    filename = "temp.mp3"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)

    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    finally:
        pygame.mixer.music.unload()
        if os.path.exists(filename):
            os.remove(filename)

# Speech recognition
def listen_once(duration=3, fs=16000):
    """
    Record a short audio clip from the microphone and transcribe it to text.

    Uses Google Speech Recognition (with Arabic language) to convert speech to text.
    Normalizes the result before returning.

    Args:
        duration (int, optional): Recording duration in seconds. Default is 3.
        fs (int, optional): Sample rate in Hz. Default is 16000.

    Returns:
        str: Normalized recognized text, or an empty string if recognition fails.
    """
    print("🎤 يَسْتَمِعُ...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    filename = "temp.wav"
    write(filename, fs, recording)

    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="ar-EG")
        print(f"🔊 قُلْتَ: {text}")
        return normalize_text(text)
    except sr.UnknownValueError:
        print("❌ لَمْ أَفْهَمِ الكَلَامَ")
        return ""
    except sr.RequestError as e:
        print(f"❌ خَطَأٌ فِي خِدْمَةِ التَّعَرُّفِ: {e}")
        return ""
    finally:
        if os.path.exists(filename):
            os.remove(filename)

# Image capture functions

def capture_Family_image():
    """
    Capture an image and save it to the 'family' folder with a custom name.

    Prompts the user to enter a filename, checks for duplicates, and saves
    the captured photo using OpenCV.

    Returns:
        str or None: File path of the saved image if successful, or None if failed.
    """
    print("📸 سَيَتِمُّ التَّصْوِيرُ بَعْدَ ٣ ثَوَانٍ... اِبْتَسِمْ 😊")
    time.sleep(3)

    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise Exception("📷 لَمْ أَتَمَكَّنْ مِنْ فَتْحِ الكَامِيرَا")

        ret, frame = cam.read()
        cam.release()

        if not ret:
            raise Exception("📷 لَمْ أَتَمَكَّنْ مِنْ التَّقَاطِ الصُّورَة")

        while True:
            img_name_input = input("📝 أَدْخِلِ اسْمَ الصُّورَةِ (بدون امتداد): ").strip()

            if img_name_input:
                img_name_clean = "".join(c for c in img_name_input if c.isalnum())
                if not img_name_clean:
                    print("❌ هَذَا الاِسْمُ غَيْرُ صَالِحٍ، جَرِّبِ اسْمًا آخَرَ.")
                    continue

                img_path = os.path.join(FOLDER_NAME, f"{img_name_clean}.png")

                if os.path.exists(img_path):
                    print("⚠️ الصُّورَةُ مَوْجُودَةٌ بِالفِعْلِ. اِخْتَرِ اسْمًا آخَرَ.")
                else:
                    break
            else:
                print("❌ لَمْ تُدْخِلِ أَيَّ اسْمٍ، جَرِّبْ مَرَّةً أُخْرَى.")

        cv2.imwrite(img_path, frame)
        print("✅ تَمَّ التَّقَاطُ الصُّورَةِ.")
        return img_path

    except Exception as e:
        print(f"❌ خَطَأٌ: {e}")
        return None


def capture_image():
    """
    Capture an image and save it to the 'captured_images' folder using a unique timestamp.

    Uses OpenCV to take the photo, waits for 3 seconds before capturing, and
    saves the image file with a timestamp-based name.

    Returns:
        str or None: File path of the saved image if successful, or None if failed.
    """
    print("📸 سَيَتِمُّ التَّصْوِيرُ بَعْدَ ٣ ثَوَانٍ... اِبْتَسِمْ 😊")
    time.sleep(3)

    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise Exception("📷 لَمْ أَتَمَكَّنْ مِنْ فَتْحِ الكَامِيرَا")

        ret, frame = cam.read()
        cam.release()

        if not ret:
            raise Exception("📷 لَمْ أَتَمَكَّنْ مِنْ التَّقَاطِ الصُّورَة")

        # نستخدم timestamp كاسم للملف لضمان التفرد
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"image_{timestamp}.png"
        img_path = os.path.join(FOLDER_NAME, img_filename)

        # نحفظ الصورة
        cv2.imwrite(img_path, frame)

        print(f"✅ تَمَّ التَّقَاطُ الصُّورَةِ وَتَخْزِينُهَا: {img_path}")
        return img_path

    except Exception as e:
        print(f"❌ خَطَأٌ: {e}")
        return None












# def start_point():
#     print("مُبْصِرُ يَقُولُ: أَهْلًا بِكَ! قُلْ: أَهْلًا مُبْصِر لِنَبْدَأ.")
#     asyncio.run(edge_speak("أَهْلًا بِكَ! أَنَا مُبْصِر. قُلْ: أَهْلًا مُبْصِر لِنَبْدَأ."))

#     while True:
#         command = listen_once(duration=3)
#         if any(word in command for word in START_COMMANDS):
#             asyncio.run(edge_speak("تَمَامٌ! يُمْكِنُكَ قَوْلُ: اِسْتَكْشِفْ المَكَان، اِلْتَقِطْ صُورَة، أَوْ شُكْرًا مُبْصِر."))
#             break
#         elif command:
#             asyncio.run(edge_speak("لَمْ أَسْمَعْ: أَهْلًا مُبْصِر، حَاوِلْ مَرَّةً أُخْرَى."))

#     while True:
#         command = listen_once(duration=3)
#         if any(word in command for word in EXPLORE_COMMANDS):
#             explore_place()
#         elif any(word in command for word in PHOTO_COMMANDS):
#             img_path = capture_image()
#             if img_path:
#                 asyncio.run(edge_speak(f"تَمَّ التَّقَاطُ الصُّورَةِ "))
#             else:
#                 asyncio.run(edge_speak("وَقَعَتْ مُشْكِلَةٌ أَثْنَاءَ التَّصْوِيرِ"))
#         elif any(word in command for word in EXIT_COMMANDS):
#             asyncio.run(edge_speak("إِلَى اللِّقَاءِ!"))
#             break
#         elif command:
#             asyncio.run(edge_speak("لَمْ أَفْهَمِ الطَّلَبَ، أَعِدِ المُحَاوَلَةَ."))





# # if __name__ == "__main__":
# #     try:
# #         start_point()
# #     finally:
# #         pygame.mixer.quit()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# import os
# import cv2
# import torch
# import sounddevice as sd
# from transformers import VitsModel, AutoTokenizer
# import speech_recognition as sr

# print("جاري تحميل النموذج...")
# model = VitsModel.from_pretrained("facebook/mms-tts-ara")
# tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ara")
# print(" تم تحميل النموذج.")

# def speak(text):
#     inputs = tokenizer(text, return_tensors="pt")
#     with torch.no_grad():
#         output = model(**inputs).waveform
#     wav = output.squeeze().cpu().numpy()
#     sd.play(wav, samplerate=model.config.sampling_rate)
#     sd.wait()
#     print(f"(مبصر قال): {text}")

# # التعرف على الكلام من الميكروفون بالعربية
# def recognize_speech():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("استمع...")
#         recognizer.adjust_for_ambient_noise(source, duration=1)
#         try:
#             audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
#             command = recognizer.recognize_google(audio, language='ar-EG')
#             print(f"Google سمع: {command}")
#             return command.lower()
#         except sr.UnknownValueError:
#             speak("لم أفهم. حاول مرة أخرى.")
#             return ""
#         except sr.RequestError as e:
#             speak("خطأ في خدمة التعرف على الكلام.")
#             return ""
#         except sr.WaitTimeoutError:
#             speak("لم أسمع أي شيء.")
#             return ""

# def ensure_family_folder():
#     if not os.path.exists("family"):
#         os.makedirs("family")

# def capture_photo():
#     ensure_family_folder()
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         speak("تعذر فتح الكاميرا.")
#         print("تعذر فتح الكاميرا.")
#         return
#     speak("انظر إلى الكاميرا. سيتم التقاط الصورة بعد ثلاث ثوان.")
#     for i in range(3,0,-1):
#         print(i)
#     ret, frame = cap.read()
#     if ret:
#         name = input("اكتب اسم الشخص: ")
#         filename = f"family/{name}.jpg"
#         cv2.imwrite(filename, frame)
#         speak(f"تم حفظ الصورة باسم {filename}")
#         print(f" صورة محفوظة في {filename}")
#     else:
#         speak("تعذر التقاط الصورة.")
#     cap.release()
#     cv2.destroyAllWindows()

# def detection_mode():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         speak("تعذر فتح الكاميرا.")
#         print("تعذر فتح الكاميرا.")
#         return
#     speak("تم تفعيل وضع الكشف. اضغط Q للخروج.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cv2.imshow("كشف", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     speak("مرحبًا، أنا مبصر. قل اهلا مبصر للبدء.")
#     while True:
#         command = recognize_speech()
#         if "اهلا مبصر" in command:
#             speak("أهلاً بك! قل كشف أو صورة أو شكرا مبصر للخروج.")
#             while True:
#                 action = recognize_speech()
#                 print(f"Google سمع: {action}")
#                 if any(word in action for word in ["كشف", "الكشف", "فتح الكاميرا", "افتح كاميرا", "افتح الكاميرا"]):
#                     detection_mode()
#                 elif any(word in action for word in ["صورة", "التصوير", "صور", "تصوير"]):
#                     capture_photo()
#                 elif any(word in action for word in ["شكرا مبصر", "شكرا يا مبصر", "خروج", "انهاء"]):
#                     speak("مع السلامة!")
#                     return
#                 elif action != "":
#                     speak("لم أفهم. قل كشف أو صورة أو شكرا مبصر.")

# if __name__ == "__main__":
#     main()


# =========================================================================================




