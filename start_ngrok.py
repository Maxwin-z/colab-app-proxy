import os
import sys
import time
from pathlib import Path
from pyngrok import ngrok, conf

# Configuration
PORT = 8000


def get_ngrok_token():
    # 1. Try .env file first
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("NGROK_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("\"'")
                if token:
                    return token

    # 2. Try Colab userdata
    try:
        from google.colab import userdata
        token = userdata.get("NGROK_TOKEN")
        if token:
            return token
    except (ImportError, Exception):
        pass

    # 3. Fall back to environment variable
    token = os.environ.get("NGROK_TOKEN")
    if token:
        return token

    print("Error: NGROK_TOKEN not found in .env, Colab userdata, or environment variables.")
    sys.exit(1)

def start_ngrok():
    # Set the auth token and region
    conf.get_default().auth_token = get_ngrok_token()
    conf.get_default().region = "ap"
    
    try:
        # Open a HTTP tunnel on the specified port
        # output is not captured here, but ngrok usually prints logs. 
        # We explicitly print the public URL.
        tunnel = ngrok.connect(PORT)
        public_url = tunnel.public_url
        print(f"==========================================")
        print(f"Ngrok Tunnel Started: {public_url}")
        print(f"==========================================")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping ngrok...")
        ngrok.kill()
        sys.exit(0)
    except Exception as e:
        print(f"Error starting ngrok: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_ngrok()
