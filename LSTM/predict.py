import torch
import argparse
import numpy as np
from LSTM_model_v4 import LSTM_model

def predict_sample(model_path, input_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Make a prediction for a single sample using the trained model
    
    Args:
        model_path: Path to the trained model checkpoint
        input_data: List of feature values (22 features)
        device: Device to run inference on
    
    Returns:
        Predicted class (1, 2, or 3) and class probabilities
    """
    # Load model
    model = LSTM_model(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Convert input to tensor
    input_tensor = torch.tensor([input_data], dtype=torch.float32).to(device)
    
    # Apply log2 to first two features
    input_tensor[0, :2] = torch.log2(input_tensor[0, :2])
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = output[0].cpu().numpy()
        predicted_class = torch.argmax(output, dim=1).item() + 1  # Add 1 to get original class labels (1, 2, 3)
    
    return predicted_class, probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using trained LSTM model")
    parser.add_argument("--model", default="best_lstm_model.pth", help="Path to trained model")
    parser.add_argument("--features", nargs='+', type=float, required=True, 
                      help="7 feature values (space separated)")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', 
                      help="Device to run inference on")
    
    args = parser.parse_args()
    
    if len(args.features) != 7:
        print(f"Error: Expected 7 features, but got {len(args.features)}")
    else:
        predicted_class, probabilities = predict_sample(args.model, args.features, args.device)
        
        print(f"Predicted class (flag): {predicted_class}")
        print(f"Class probabilities: {probabilities}")
        print(f"Class 1: {probabilities[0]:.4f}, Class 2: {probabilities[1]:.4f}, Class 3: {probabilities[2]:.4f}")