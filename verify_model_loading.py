import os
import pickle

def load_model(model_path):
    """Function to load a model from a given path."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Define paths
base_path = '/home/bradk/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/model'
encoder_path = os.path.join(base_path, 'encoder.pkl')
model_path = os.path.join(base_path, 'model.pkl')

# Load the encoder
print(f'Loading encoder from: {encoder_path}')
try:
    encoder = load_model(encoder_path)
    print('Encoder loaded successfully.')
except Exception as e:
    print(f'Error loading encoder: {str(e)}')

# Load the model
print(f'Loading model from: {model_path}')
try:
    model = load_model(model_path)
    print('Model loaded successfully.')
except Exception as e:
    print(f'Error loading model: {str(e)}')

# Verify attributes (if applicable)
if encoder is not None:
    print(f'Encoder attributes: {dir(encoder)}')  # List attributes of the encoder

if model is not None:
    print(f'Model attributes: {dir(model)}')  # List attributes of the model
