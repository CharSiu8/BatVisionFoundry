from unittest import result
import gradio as gr
import requests
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from openai import OpenAI

# load environment variables
load_dotenv()

print("ENV CHECK:")
print(f"Endpoint length: {len(os.getenv('CUSTOM_VISION_ENDPOINT', ''))}")
print(f"Key length: {len(os.getenv('CUSTOM_VISION_KEY', ''))}")

# config: endpoint, key
CUSTOM_VISION_ENDPOINT = os.getenv("CUSTOM_VISION_ENDPOINT")
CUSTOM_VISION_KEY = os.getenv("CUSTOM_VISION_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CUSTOM_VISION_PROJECT_ID = os.getenv("CUSTOM_VISION_PROJECT_ID")
iteration_name = os.getenv("CUSTOM_VISION_PUBLISHED_NAME")
client = OpenAI(api_key=OPENAI_API_KEY)

if not CUSTOM_VISION_ENDPOINT:
    print("Error: CUSTOM_VISION_ENDPOINT not set in .env")
    exit(1)
if not CUSTOM_VISION_KEY:
    print("Error: CUSTOM_VISION_KEY not set in .env")
    exit(1)
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not set in .env")
    exit(1)

print("Endpoint:", CUSTOM_VISION_ENDPOINT)
print("Key:", CUSTOM_VISION_KEY[:10] + "...")

# Agentic AI feature that uses LLM in order to pull a random quote matched to that specific Batman/lookalike
def get_quote(actor):
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = [
            {"role": "system", "content": "You return one iconic quote from the specified character. For Batman actors (Affleck, Bale, Pattinson), return a quote from their Batman movies. For Nite Owl, return a quote from Watchmen. For Darkwing, return a quote from Invincible. Just the quote, no attribution or extra text."},
            {"role": "user", "content": f"Give me an iconic quote from {actor}'s Batman movies."}
            ],
            max_tokens =100,

    )
    return response.choices[0].message.content

# Agentic AI Features to match image of Batman/lookalike to movie by suit and pull box office data 
# Feature 1 match image to movie/show
def get_movie(image_path, actor):
    import base64
    
    # open image and convert to base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
        
    # call GPT-4o with vision
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""This is {actor} as Batman or a Batman lookalike. 
                    Based on the suit design, cowl shape, and visual style, which specific movie is this from? 
                    Options for reference: Affleck (Batman v Superman, Justice League), Bale (Batman Begins, 
                    The Dark Knight, The Dark Knight Rises), Pattinson (The Batman), Nite Owl (Watchmen), 
                    Darkwing (Invincible). Reply with only the exact movie title."""},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                        }
                    }
                ]
            }
        ],
        max_tokens=50
            )

    return response.choices[0].message.content
                
# Feature 2- Movie details
def get_movie_details(movie):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Return movie details in this exact format: This Batman fought: [villian name] | Box Office: [amount]"},
            {"role": "user", "content": f"For the movie {movie}, who is the main villain and what was the box office total?"}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content

# function: predict(image)
def predict(image):
    # create a BytesIO buffer
    buffer = BytesIO()
    
    # use PIL Image to open the image
    img = Image.open(image)
    
    # save image to buffer as JPEG
    img.save(buffer, format="JPEG")
    
    # get bytes using buffer.getvalue()
    image_bytes = buffer.getvalue()
    
    # send POST request to Custom Vision API
    headers = {
        "Prediction-Key": CUSTOM_VISION_KEY,
        "Content-Type": "application/octet-stream"
    }
    prediction_url = f"{CUSTOM_VISION_ENDPOINT}/customvision/v3.0/Prediction/{CUSTOM_VISION_PROJECT_ID}/classify/iterations/{iteration_name}/image"
    print("DEBUG - Full URL:", prediction_url)
    print("DEBUG - iteration_name:", iteration_name)
    response = requests.post(prediction_url, headers=headers, data=image_bytes)
    
    # parse response
    result = response.json()
    print(result)
    print("DEBUG - API Response:", result)
    print("DEBUG - Endpoint:", CUSTOM_VISION_ENDPOINT)
    print("DEBUG - Project ID:", CUSTOM_VISION_PROJECT_ID)
    print("DEBUG - API Response:", result)
    predictions = result.get("predictions", [])
    if not predictions:
        return "Error: No predictions returned. Check Custom Vision model."
    top = max(predictions, key=lambda x: x["probability"])
    
    # return top prediction + confidence
    name = top["tagName"].capitalize()
    confidence = top["probability"] * 100
    # Agentic AI Feature 3, return quote
    movie = get_movie(image, name)
    details = get_movie_details(movie)
    quote = get_quote(name)

    return f"{name} ({confidence:.1f}% confidence)\nMovie: {movie}\n{details}\n\n{quote}"

# interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", height=200),
    outputs=gr.Text(label="Prediction"),
    title="BatVision",
    description="Upload an image of Batman or a Batman lookalike to identify the actor. This model accepts: Ben Affleck, Christian Bale, Robert Pattinson, Nite Owl (Watchmen) & Darkwing (Invincible).",

examples=[
    ["UI_test_images/affleck_test1.jpg"],
    ["UI_test_images/affleck_test2.jpg"],
    ["UI_test_images/pattinson_test1.jpg"],
    ["UI_test_images/pattinson_test2.jpg"],
    ["UI_test_images/bale_test1.jpg"],
    ["UI_test_images/bale_test2.jpg"],
    ["UI_test_images/darkwing_test1.jpg"],
    ["UI_test_images/darkwing_test2.jpg"],
    ["UI_test_images/niteowl_test1.jpg"],
    ["UI_test_images/niteowl_test2.jpg"]
]
)

# launch
# launches on both azure and gradio/Hugging face
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
