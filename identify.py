import os
import sys
import requests
import ollama


MODEL = "minicpm-v4.6"
OLLAMA_URL = "http://127.0.0.1:11434"


def check_ollama():

    try:
        r = requests.get(
            f"{OLLAMA_URL}/api/version",
            timeout=5
        )

        if r.status_code == 200:
            print("Ollama:", r.json().get("version"))
            return True

    except Exception:
        pass

    return False


def check_model():

    try:

        result = ollama.list()

        models = result.get("models", [])

        for model in models:

            name = model.get("model", "")

            if name.startswith(MODEL):
                return True

    except Exception as e:

        print("Could not check models:", e)

    return False


def download_model():

    print(f"Downloading {MODEL}...")

    try:

        ollama.pull(MODEL)

        print("Model ready.")

    except Exception as e:

        print("Model download failed:")
        print(e)

        sys.exit(1)


def analyze_image(image_path):

    if not os.path.exists(image_path):

        print("Image does not exist:")
        print(image_path)

        sys.exit(1)  

    print()
    print("Analyzing image...")
    print()

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": """
                    Identify the unique objects visible in this image.

                    Return ONLY a dot-separated list of object names.

                    Rules:
                    - Each object only once
                    - No duplicates
                    - No counts
                    - No descriptions
                    - No explanations
                    - Use short common object names
                    - Do not hallucinate

                    Example:
                    person. chair. table. laptop. phone. bottle.
                    """,
                    "images": [image_path]
                }
            ],
            options={
                "temperature": 0,
                "num_ctx": 2048,
                "num_predict": 100,
            },
            keep_alive="30m",
        )


        answer = response["message"]["content"]

        print("=" * 70)
        print("IMAGE DESCRIPTION")
        print("=" * 70)
        print()

        print(answer)

        print()
        print("=" * 70)

        return answer

    except Exception as e:

        print()
        print("VISION ERROR")
        print("=" * 70)
        print(e)
        print("=" * 70)

        sys.exit(1)


def main():

    if len(sys.argv) != 2:

        print("Usage:")
        print()
        print("python script.py poster_image.jpg")

        sys.exit(1)

    image_path = sys.argv[1]

    print("=" * 70)
    print("LOCAL AI IMAGE ANALYZER")
    print("=" * 70)

    print()
    print("Image :", image_path)
    print("Model :", MODEL)

    if not check_ollama():

        print()
        print("ERROR: Ollama is not running.")
        print()
        print("Start it with:")
        print("ollama serve")

        sys.exit(1)

    if not check_model():

        download_model()

    else:

        print("Model already installed:", MODEL)

    analyze_image(image_path)


if __name__ == "__main__":
    main()
