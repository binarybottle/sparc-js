import pickle
import numpy as np
import json
import os

def convert_linear_model_to_json(pkl_path, json_path):
    """
    Convert the pickle linear model to JSON format for use in JavaScript.
    
    Args:
        pkl_path: Path to the pickle file containing the linear model
        json_path: Path to save the JSON output
    """
    # Load the pickle file
    print(f"Loading linear model from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        linear_model = pickle.load(f)
    
    # Extract weights and biases
    weights = linear_model.coef_.tolist()
    biases = linear_model.intercept_.tolist()
    
    # Create a dictionary to store the model parameters
    model_json = {
        'weights': weights,
        'biases': biases,
        'input_dim': linear_model.coef_.shape[1],
        'output_dim': linear_model.coef_.shape[0]
    }
    
    # Add metadata
    model_json['metadata'] = {
        'description': 'Linear projection model for SPARC articulatory features',
        'features': ['ul_x', 'ul_y', 'll_x', 'll_y', 'li_x', 'li_y', 
                     'tt_x', 'tt_y', 'tb_x', 'tb_y', 'td_x', 'td_y'],
        'created_from': os.path.basename(pkl_path)
    }
    
    # Save to JSON
    print(f"Saving model to {json_path}")
    with open(json_path, 'w') as f:
        json.dump(model_json, f)
    
    # Print model details
    print(f"Model converted successfully:")
    print(f"  - Input dimensions: {model_json['input_dim']}")
    print(f"  - Output dimensions: {model_json['output_dim']}")
    print(f"  - JSON file size: {os.path.getsize(json_path) / (1024*1024):.2f} MB")
    
    return model_json

if __name__ == "__main__":
    # Path to your pickle file
    pkl_path = "wavlm_large-9_cut-10_mngu_linear.pkl"
    
    # Path to save the JSON file
    json_path = "wavlm_linear_model.json"
    
    # Convert the model
    convert_linear_model_to_json(pkl_path, json_path)