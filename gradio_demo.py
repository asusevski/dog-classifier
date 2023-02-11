import gradio as gr
from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ColorJitter, ToTensor, RandomPerspective
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


with open("./data/dogs_labels.txt", "r") as f:
    labels = f.read().split('\n')


num_labels = len(labels)
id2label = {str(i): c for i, c in enumerate(labels)}


def classify_image(inp):
    # Load model
    model = AutoModelForImageClassification.from_pretrained(pretrained_model_name_or_path="asusevski/vit-dog-classifier")

    # Preprocess
    model_preprocessor_name = "google/vit-base-patch16-224"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_preprocessor_name)

    inp = np.array(inp.convert('RGB'))
    
    inp = torch.tensor(feature_extractor(images=inp)['pixel_values'])
    preds = model(inp)['logits']
    preds = torch.flatten(preds)
    preds = nn.functional.softmax(preds, dim=0)
    confidences = {labels[i]: preds[i].item() for i in range(num_labels)}
    return confidences


gr.Interface(fn=classify_image, 
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    examples=['./data/dogs/train/Yorkie/01.jpg', './data/dogs/train/Yorkie/02.jpg'],
    title="Dog Image",
).launch()
