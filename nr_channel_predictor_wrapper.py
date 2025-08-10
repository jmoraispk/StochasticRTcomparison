import os
import numpy as np
import torch
import nr_channel_predictor as torch_module

###################################################################
#                   Neural Network Model
###################################################################
def construct_model(num_tx_antennas: int, hidden_size: int, num_layers: int, filename: str=""):
    """
    Constructs and optionally loads a PyTorch model for channel prediction.

    This function creates a GRU-based channel predictor model with the specified
    architecture parameters. If a filename is provided, it also loads the model
    weights from the specified file.

    Args:
        num_tx_antennas (int): Number of transmit antennas. The input and output features
                               are channel predictions for these antennas, represented as
                               in-phase (I) and quadrature (Q) components.
        hidden_size (int): Number of features in the hidden state of the GRU.
        num_layers (int): Number of recurrent layers in the GRU.
        filename (str, optional): File name for saved model weights. If provided, the
                                  weights are loaded into the model.

    Returns:
        torch.nn.Module: A PyTorch model for channel prediction. If a filename is provided,
                         the returned model includes the loaded weights.
    """

    # Input and the output features are channel predictions for the  Tx antennas. 
    # Since the values are complex, feed them in as I and Q
    chan_predictor = torch_module.ChannelPredictorGRU(input_size=num_tx_antennas*2, hidden_size=hidden_size, 
                                            output_size=num_tx_antennas*2, num_layers=num_layers)

    if (filename):
        model = load_model_weights(chan_predictor, filename)
    return chan_predictor

###################################################################
#                   Train
###################################################################
def train(chan_predictor, 
          x_train: np.ndarray, y_train: np.ndarray, 
          x_val: np.ndarray, y_val: np.ndarray, 
          initial_learning_rate: float, batch_size: int, num_epochs: int, 
          validation_freq: int = 1, verbose: bool=False, patience: int = 10):
    """
    Trains a channel predictor PyTorch model using offline training.

    This function performs offline training of the provided channel predictor
    model using in-memory data. The model is trained with the specified hyperparameters,
    and training progress can be optionally displayed.

    Args:
        chan_predictor (torch.nn.Module): The PyTorch model to be trained.
        x_train (np.ndarray): Input data for training.
        y_train (np.ndarray): Target output data for training.
        x_val (np.ndarray): Input data for validation.
        y_val (np.ndarray): Target output data for validation.
        initial_learning_rate (float): The initial learning rate for the optimizer.
        batch_size (int): The number of samples per training batch.
        num_epochs (int): The total number of epochs for training.
        validation_freq (int): Frequency of validation checks during training, measured in epochs.
        verbose (bool, optional): If True, displays training progress and information.
        patience (int, optional): Number of epochs to wait for improvement before early stopping.

    Returns:
        tuple: A tuple containing:
            - chan_predictor (torch.nn.Module): The trained model.
            - training_loss (list): A list of training loss values recorded per iteration or epoch.
            - validation_loss (list): A list of validation loss values recorded every `validation_freq` epochs.
            - elapsed_time (float): The total elapsed time for the training process.
    """
    
    device = select_device(verbose=True)
  
    # Initialize the channel prediction manager
    chanpre = torch_module.ChannelPredictorManager(chan_predictor, lr=initial_learning_rate, 
                                               device=device, verbose=verbose)

    # Prepare data loaders for training and validation
    train_loader, x_val_tensor, y_val_tensor = chanpre.prepare_data(
        x_train, y_train, 
        x_val, y_val, 
        batch_size)

    # Train the model
    training_loss, validation_loss, elapsed_time = chanpre.train(
        train_loader, x_val_tensor, y_val_tensor,
        num_epochs=num_epochs, patience=patience)
    
    # Return a list containing the trained model and training metrics
    # This list appears as cell array in MATLAB
    return chan_predictor, training_loss, validation_loss, elapsed_time
  
###################################################################
#                   Predict
###################################################################
def predict(chan_predictor, x_data):
    """
    Generates predictions using a trained PyTorch model and input data.

    This function uses the provided channel predictor model to make predictions
    on the input data. The model is set to evaluation mode, and computations
    are performed without tracking gradients to improve efficiency.

    Args:
        chan_predictor (torch.nn.Module): The trained PyTorch model used for making predictions.
        x_data (np.ndarray): The input data for which predictions are to be generated.

    Returns:
        np.ndarray: The predicted output data.
    """
    
    # Select the appropriate device (CPU or GPU) for training
    device = select_device()

    # Convert input data to a PyTorch tensor and move it to the specified device
    x_tensor = torch.from_numpy(x_data).float().to(device)
    
    # Set the model to evaluation mode
    chan_predictor.eval().to(device)
    
    # Disable gradient calculation
    with torch.no_grad():
        # Perform the forward pass to get predictions
        outputs = chan_predictor(x_tensor)
    
    # Move the predictions back to the CPU and convert them to a numpy array
    predictions = outputs.cpu().numpy()
    
    return predictions

###################################################################
#                   Save model
###################################################################
def save_model_weights(chan_predictor, filename: str):
    """
    Saves the state dictionary of a PyTorch model to a file.

    This function saves the model's weights to a specified file, ensuring that
    the file has a '.pth' extension. The model is moved to the CPU before saving
    to ensure compatibility across different devices.

    Args:
        chan_predictor (torch.nn.Module): The PyTorch model whose weights are to be saved.
        filename (str): The desired filename for saving the model's state dictionary.
                        If the filename does not end with '.pth', the extension will be appended.

    Returns:
        str: The actual filename used for saving the model's state dictionary, including the '.pth' extension.
    """
    # Ensure the filename has a .pth extension
    base, _ = os.path.splitext(filename)
    new_filename = base + '.pth'

    # Move the model to CPU for saving
    chan_predictor.to("cpu")

    # Save the model state dictionary and metadata to the specified file
    torch.save({
        'model_state_dict': chan_predictor.state_dict()
    }, new_filename)

    # Return the filename used for saving
    return new_filename

###################################################################
#                   Load model weights
###################################################################
def load_model_weights(chan_predictor, filename: str):
    """
    Loads a state dictionary into a PyTorch model from a specified file.

    This function loads the model weights from a file and updates the given
    PyTorch model with these weights. It ensures the file has the correct
    extension and exists before attempting to load.

    Args:
        chan_predictor (torch.nn.Module): The PyTorch model to be updated with new weights.
        filename (str): The path to the file containing the saved model state dictionary.
                        The file must have a '.pth' extension.

    Returns:
        torch.nn.Module: The PyTorch model updated with the new weights.

    Raises:
        ValueError: If the file does not have a '.pth' extension.
        FileNotFoundError: If the specified file does not exist.
    """
    
    # Check if the filename has a .pth extension
    if not filename.endswith('.pth'):
        raise ValueError("The file must have a .pth extension.")

    # Check if the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    
    # Load the checkpoint from the specified file
    device = select_device()
    checkpoint = torch.load(filename, map_location=torch.device(device), weights_only=False)
    
    # Load the model's state dictionary from the checkpoint
    chan_predictor.load_state_dict(checkpoint['model_state_dict'])
    
    # Call flatten_parameters to optimize the model's internal state
    chan_predictor.flatten_parameters()
    
    # Return the updated instance
    return chan_predictor

###################################################################
#                   Model Information
###################################################################
def info(chan_predictor):
    """
    Prints detailed information about a PyTorch model.

    This function outputs the architecture and the total number of parameters
    of the provided PyTorch model. 

    Args:
        chan_predictor (torch.nn.Module): The PyTorch model to be analyzed.
    """
    print("Model architecture:")
    print(chan_predictor)
    
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in chan_predictor.parameters())
    print(f"\nTotal number of parameters: {total_params}")

###################################################################
#                   Helpers
###################################################################
def select_device(verbose: bool = False) -> torch.device:
    """
    Selects the computational device for PyTorch operations.

    This function chooses the appropriate computational device (GPU or CPU)
    based on the availability of CUDA.

    Args:
        verbose (bool): If True, prints information about the selected device.

    Returns:
        torch.device: The selected device, either GPU ('cuda') if available, or CPU ('cpu').
    """
    # Determine the device based on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Verbose output for debugging and information
    if verbose:
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert bytes to GB
            print(f"Selected device: GPU ({gpu_name}, {total_memory:.2f} GB)")
        else:
            print("Selected device: CPU")

    return device