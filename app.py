import os
import streamlit as st
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import h2o
from h2o.automl import H2OAutoML
from transformers import BertTokenizer, BertForSequenceClassification, Wav2Vec2Tokenizer, Wav2Vec2ForSequenceClassification
import librosa
import time

# Function to load data
def load_data(file_path):
    """Load data based on file extension"""
    if os.path.isdir(file_path):
        image_files = [os.path.join(file_path, f) for f in os.listdir(file_path)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if image_files:
            return image_files, 'unstructured'
        else:
            raise ValueError("No image files found in the directory.")
    else:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.csv':
            return pd.read_csv(file_path), 'structured'
        elif file_ext in ['.jpg', '.png', '.jpeg']:
            return file_path, 'unstructured'  # Return path for image files
        elif file_ext in ['.wav', '.mp3']:
            return file_path, 'unstructured'  # Return path for audio files
        elif file_ext in ['.txt']:
            return pd.read_csv(file_path, header=None, names=['text']), 'structured'
        else:
            raise ValueError("Unsupported file type")

# Function to analyze target variable
def analyze_target_variable(df, target_column):
    """Determine if the column is suitable for classification or regression"""
    dtype = df[target_column].dtype
    unique_values = df[target_column].nunique()

    if dtype in ['object', 'category']:
        return 'classification', unique_values
    elif dtype in ['int64', 'float64']:
        if unique_values < 20:
            return 'classification', unique_values
        else:
            variance = df[target_column].var()
            return 'regression', variance
    else:
        return 'unknown', None

# Function to select target variable
def select_target_variable(df):
    """Select the most suitable target variable"""
    candidates = []

    for col in df.columns:
        task_type, measure = analyze_target_variable(df, col)
        candidates.append((col, task_type, measure))

    # Sort candidates by task type and measure
    sorted_candidates = sorted(candidates, key=lambda x: (x[1], -x[2] if x[1] == 'regression' else x[2]))

    # Select the best candidate based on sorted order
    for col, task_type, measure in sorted_candidates:
        if task_type != 'unknown':
            return col, task_type

    return None, 'unknown'

# Function to process text data with BERT
def process_text_with_bert(texts):
    """Process text data with BERT model"""
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    return predictions.numpy()

# Function to process image data
def process_image(image_path, categories=None):
    """Process image data and visualize results, optionally filtering by specified categories."""
    if not os.path.isfile("imagenet_classes.txt"):
        raise FileNotFoundError("imagenet_classes.txt file not found. Please ensure it is downloaded and in the correct directory.")

    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)

    model = models.resnet50(pretrained=True)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)

    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    if categories:
        filtered_top5_catid = []
        filtered_top5_prob = []
        for catid, prob in zip(top5_catid, top5_prob):
            label = labels[catid.item()]
            if label in categories:
                filtered_top5_catid.append(catid.item())
                filtered_top5_prob.append(prob.item())
    else:
        filtered_top5_catid = top5_catid.tolist()
        filtered_top5_prob = top5_prob.tolist()

    # Display the image with predictions in Streamlit
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Top-5 Predictions and Probabilities:")
    for i, (catid, prob) in enumerate(zip(filtered_top5_catid, filtered_top5_prob)):
        label = labels[catid]
        st.write(f"{i + 1}: {label} - {prob:.4%}")

    return filtered_top5_catid, filtered_top5_prob

# Function to process audio data
def process_audio(audio_path):
    """Process audio data"""
    model_name = 'facebook/wav2vec2-large-xlsr-53'
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

    y, sr = librosa.load(audio_path, sr=16000)
    inputs = tokenizer(y, return_tensors="pt", padding="longest")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    return predictions.numpy()

def process_folder(folder_path, categories=None):
    """Process all images in a folder"""
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        st.error("No image files found in the directory.")
        return
    
    for image_file in image_files:
        st.write(f"Processing {image_file}...")
        top5_catid, top5_prob = process_image(image_file, categories)
        st.write("Top-5 Predictions and Probabilities:")
        for i in range(len(top5_catid)):
            st.write(f"{i + 1}: Category {top5_catid[i]} with probability {top5_prob[i]:.4f}")

# Updated main function to display image and predictions
def main():
    st.title("AutoML App")

    # Initialize folder_path to None
    folder_path = None
    uploaded_file= None
    # Choose between file or folder
    option = st.radio("Select input type", ('File', 'Folder'))

    if option == 'File':
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "jpg", "png", "jpeg", "wav", "mp3", "txt"])
    elif option == 'Folder':
        folder_path = st.text_input("Or enter the path to a folder of images")

    if uploaded_file:
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        data, data_type = load_data(os.path.join("temp", uploaded_file.name))

        st.write(f"Detected data type: {data_type}")

        if data_type == 'structured':
            with st.spinner('Processing...'):
                h2o.init()
                target, task_type = select_target_variable(data)
                st.write(f"Selected Target Variable: {target}")
                st.write(f"Task Type: {task_type}")

                hf = h2o.H2OFrame(data)
                features = hf.columns
                features.remove(target)

                aml = H2OAutoML(max_runtime_secs=3600, max_models=20, seed=42)
                aml.train(x=features, y=target, training_frame=hf)

                lb = aml.leaderboard
                best_model = aml.leader
                st.success("Processing complete!")
                st.write("Best Model:", best_model)

        elif data_type == 'unstructured':
            if isinstance(data, list) and data[0].endswith(('.jpg', '.png', '.jpeg')):
                for image_path in data:
                    st.write(f"Processing {image_path}...")
                    top5_catid, top5_prob = process_image(image_path)
                    st.write("Top-5 Predictions and Probabilities:")
                    for i in range(len(top5_catid)):
                        st.write(f"{i + 1}: Category {top5_catid[i]} with probability {top5_prob[i]:.4f}")

            elif uploaded_file.name.endswith(('.wav', '.mp3')):
                predictions = process_audio(uploaded_file.name)
                st.write("Audio Predictions:", predictions)

            elif uploaded_file.name.endswith('.txt'):
                texts = data['text'].tolist()
                predictions = process_text_with_bert(texts)
                st.write("BERT Predictions:", predictions)

    # Ensure folder_path is checked only if it's valid
    elif folder_path and os.path.isdir(folder_path):
        process_folder(folder_path)

if __name__ == "__main__":
    if not os.path.exists("temp"):
        os.makedirs("temp")
    main()
