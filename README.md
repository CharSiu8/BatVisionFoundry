# BatVisionAzure
BatVision Deployed on Azure

# BatVision

AI-powered image classifier that identifies Batman actors (Affleck, Bale, Pattinson) and distinguishes from similar masked characters (Nite Owl, Darkwing). Built with Azure cloud services.

## LIVE DEMO

http://batvision.a4dyasbea5bxc7gd.canadaeast.azurecontainer.io:8000/

*Please note that the Azure version is off between 7pm-9am ET to save 58% in monthly costs.*

## See also the same application deployed on Hugging Face:
https://huggingface.co/spaces/CharSiu8/BatVision

## Features
- **Image Classification:** Upload an image → identifies which Batman actor
- **Masked Face Recognition:** Model trained on challenging mask-heavy images, forcing recognition of subtle facial features (jaw, chin, mouth)
- **Similarity Detection:** Distinguishes Batman from lookalike characters (Nite Owl from Watchmen, Darkwing from Invincible)
- **Agentic Quote Generation:** GPT generates a random iconic quote from that specific Batman's movies
- **Agentic Movie Detection:** GPT-4o Vision analyzes the suit design to identify the exact movie/show
- **Agentic Villain & Box Office Lookup:** GPT fetches the main villain and box office data for the detected movie

## Azure Integrations

- **Azure Blob Storage:** Cloud hosting for training images
- **Azure Custom Vision:** Model training and prediction API
- **Azure Resource Management:** Organized under `batvision-rg` resource group

## Tech Stack

- Python
- DuckDuckGo Search (data collection)
- Azure Custom Vision (model training + prediction)
- Azure Blob Storage (image hosting)
- Gradio (UI)
- HuggingFace Spaces (deployment)
- OpenAI API 
- Docker Containerizes the application for cloud deployment.

## Data Structure
```
/data
  /raw
    /affleck
    /bale
    /pattinson
    /niteowl
    /darkwing
  /processed
  manifest.json
```

## Project Scripts

- `collect_images.py` — Scrapes images via DuckDuckGo, updates manifest
- `preprocess_images.py` — Resizes to 224x224, converts to RGB/JPG
- `upload_to_azure.py` — Uploads processed images to Azure Blob Storage
- `app.py` — Gradio interface calling Azure Custom Vision API

## Quick Start
LIVE DEMO: https://huggingface.co/spaces/CharSiu8/BatVision

OR be complicated if you want
1. Clone repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` with API keys (see `.env.example`)
4. Run: `python app.py`

## Environment Variables
```
AZURE_STORAGE_CONNECTION_STRING=your_blob_storage_connection_string
CUSTOM_VISION_ENDPOINT=your_custom_vision_prediction_url
CUSTOM_VISION_KEY=your_custom_vision_prediction_key
```

## How It Works

1. User uploads image of masked character
2. Image sent to Azure Custom Vision API
3. Model returns prediction + confidence score
4. Result displayed in Gradio interface

## Model Performance

- Precision: 90.4%
- Recall: 90.4%
- Average Precision: 95.7%

## License

All Rights Reserved 2026 — Employers and Recruiters may clone and access to test
