import base64
import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')

url = 'https://api.sarvam.ai/text-to-speech'

headers = {'API-Subscription-Key': api_key}

def main_TTS(text_data):
    payload = {
    "inputs": [text_data],
    "target_language_code": "hi-IN",
    "speaker": "meera",
    "pitch": 0,
    "pace": 1.65,
    "loudness": 1.5,
    "speech_sample_rate": 8000,
    "enable_preprocessing": True,
    "model": "bulbul:v1"
	}
    response = requests.post(url, headers=headers, json=payload)
    json_data = response.json()
    audio_data = json_data.get("audios")
    return audio_data


if __name__=="__main__":
    main_TTS("What is NLP")
