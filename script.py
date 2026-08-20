import re
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

MODEL_NAME = "Salesforce/blip-image-captioning-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained(MODEL_NAME)

model = BlipForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()


# ---------------------------------------------------------
# OBJECT NAME GENERATOR
# ---------------------------------------------------------

def get_object_names(image_path: str):
    """
    Automatically analyzes an image and returns candidate
    object names that can be passed to Grounding DINO.

    Example:

        [
            "person",
            "car",
            "tree",
            "shoe",
            "logo"
        ]

    Grounding DINO prompt:

        person . car . tree . shoe . logo .
    """

    image = Image.open(image_path).convert("RGB")

    # Keep memory usage low
    image.thumbnail((1024, 1024))

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=3
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    print("Image description:")
    print(caption)

    # -----------------------------------------------------
    # Extract words from caption
    # -----------------------------------------------------

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9_-]*\b",
        caption.lower()
    )

    # Words that aren't useful Grounding-DINO classes
    stop_words = {
        "a",
        "an",
        "the",
        "and",
        "with",
        "on",
        "in",
        "at",
        "of",
        "to",
        "is",
        "are",
        "this",
        "that",
        "there",
        "image",
        "photo",
        "picture",
        "showing",
        "shows",
        "looking",
        "standing",
        "sitting",
        "wearing",
        "next",
        "near",
        "front",
        "back",
        "side",
    }

    objects = []

    for word in words:

        if word in stop_words:
            continue

        if word not in objects:
            objects.append(word)

    return objects


# ---------------------------------------------------------
# GROUNDING DINO PROMPT
# ---------------------------------------------------------

def create_dino_prompt(image_path: str):

    objects = get_object_names(image_path)

    prompt = " . ".join(objects) + " ."

    return objects, prompt


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    image_path = "poster_image.jpg"

    objects, dino_prompt = create_dino_prompt(image_path)

    print("\nObjects:")
    for obj in objects:
        print("-", obj)

    print("\nGrounding DINO prompt:")
    print(dino_prompt)