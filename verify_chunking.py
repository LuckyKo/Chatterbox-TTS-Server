
import requests
import json
import time

API_URL = "http://localhost:8884/v1/audio/speech"

def test_long_text_chunking():
    # A text long enough to trigger chunking (assuming default chunk_size=120)
    # 120 * 1.5 = 180 chars limit. We'll use ~300 chars.
    long_text = "This is the first sentence that is reasonably long. " * 10
    
    payload = {
        "model": "chatterbox-turbo",
        "input": long_text,
        "voice": "am_Ryan.wav", # Ensure this voice exists or use a safe default
        "response_format": "mp3",
        "speed": 1.0
    }
    
    print(f"Sending request with {len(long_text)} characters...")
    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"Success! Status 200. Time: {end_time - start_time:.2f}s")
            print(f"Content length: {len(response.content)} bytes")
            # We can't verify splitting happened from client side easily without checking server logs,
            # but a 200 OK on a long text means it didn't crash.
            return True
        else:
            print(f"Failed: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is it running?")
        return False

if __name__ == "__main__":
    test_long_text_chunking()
