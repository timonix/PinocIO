import pyttsx3

# initialize Text-to-speech engine
engine = pyttsx3.init()

rate = engine.getProperty('rate')  # getting details of current speaking rate
print(rate)  # printing current voice rate
engine.setProperty('rate', 125)  # setting up new voice rate


def set_volume(new_volume):
    volume = engine.getProperty('volume')  # getting to know current volume level (min=0 and max=1)
    if 1.0 >= new_volume >= 0:
        engine.setProperty('volume', new_volume)  # setting up volume level  between 0 and 1


def speak(voice_line):
    # """VOICE"""
    try:
        voices = engine.getProperty('voices')  # getting details of current voice
        engine.setProperty('voice', voices[60].id)  # changing index, changes voices. 1 for female
        engine.say(voice_line)
        engine.runAndWait()
    except:
        print("Could not find voice")
