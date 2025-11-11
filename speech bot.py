
import speech_recognition as sr
import pyttsx3
import random

# Initialize speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Sample tasks
tasks = {
    "open youtube": "Opening YouTube...",
    "set reminder": "Reminder set for 7 PM.",
}

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command.lower()
        except Exception as e:
            print("Sorry, I didn't catch that.")
            return ""

def speak(text):
    engine.say(text)
    engine.runAndWait()

def process_command(command):
    if "open youtube" in command:
        speak(tasks["open youtube"])
        # Add code here to open YouTube on Android/iOS (e.g., via Tasker)
    elif "set reminder" in command:
        speak(tasks["set reminder"])
        # Integrate reminder functionality here
    else:
        speak("I'm not sure how to do that.")

def start_bot():
    speak("Hello! How can I assist you today?")
    while True:
        user_input = listen()
        if "exit" in user_input:
            speak("Goodbye!")
            break
        process_command(user_input)

# Start the bot
start_bot()