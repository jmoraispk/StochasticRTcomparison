import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Ignore warnings especialy "NumPy array is not writeable"
warnings.filterwarnings("ignore")

class ChannelPredictionMetadata:
    def __init__(self, type, horizon, description):
        self.type = type
        self.horizon = horizon, 
        self.description = description

class NMSELoss(nn.Module):
    """
    Normalized Mean Squared Error (NMSE) Loss.

    This loss function computes the normalized mean squared error between the predicted and target values.
    It is particularly useful for regression tasks where the scale of the target values varies significantly.

    Attributes:
    - epsilon: float, a small constant added to the denominator to prevent division by zero.
    """

    def __init__(self, epsilon=1e-10):
        """
        Initializes the NMSELoss with a specified epsilon value.

        Parameters:
        - epsilon: float, a small constant to prevent division by zero in the normalization term.
        """
        super(NMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predicted, target):
        """
        Computes the NMSE loss between the predicted and target values.

        Parameters:
        - predicted: torch.Tensor, the predicted values.
        - target: torch.Tensor, the true target values.

        Returns:
        - nmse_loss: torch.Tensor, the computed NMSE loss.
        """
        mse_loss = torch.mean((predicted - target) ** 2)  # Mean squared error
        mst = torch.mean(target ** 2) + self.epsilon  # Mean squared target with epsilon for stability
        nmse_loss = mse_loss / mst  # Normalized mean squared error
        return nmse_loss

class ChannelPredictorGRU(nn.Module):
    """
    A GRU-based neural network model for channel prediction.

    This model consists of a Gated Recurrent Unit (GRU) layer followed by a batch normalization layer
    and a fully connected linear layer. It is designed to predict output channels based on input sequences.

    Attributes:
    - gru: nn.GRU, the GRU layer for sequential data processing.
    - batch_norm: nn.BatchNorm1d, the batch normalization layer applied after the GRU.
    - fc: nn.Linear, the fully connected layer for producing the final output.
    """

    def __init__(self, input_size=4, hidden_size=64, output_size=4, num_layers=2):
        """
        Initializes the ChannelPredictorGRU model with specified parameters.

        Parameters:
        - input_size: int, the number of expected features in the input.
        - hidden_size: int, the number of features in the hidden state of the GRU.
        - output_size: int, the number of features in the output.
        - num_layers: int, the number of recurrent layers in the GRU.
        """
        super(ChannelPredictorGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        - x: torch.Tensor, input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
        - out: torch.Tensor, output tensor of shape (batch_size, output_size).
        """
        out, _ = self.gru(x)  # Pass input through GRU layer
        out = self.batch_norm(out[:, -1, :])  # Apply batch normalization to the last time step
        out = self.fc(out)  # Pass through the fully connected layer
        return out
    
    def flatten_parameters(self):
        """
        Flattens the parameters of the GRU layer to optimize memory usage.
        """
        self.gru.flatten_parameters()

class ChannelPredictorManager:
    """
    Train, test, run channel prediction using a GRU-based neural network.

    This class encapsulates the model architecture, loss function, optimizer, and learning rate scheduler
    for training a GRU-based channel prediction model.

    Attributes:
    - model: nn.Module, the GRU-based prediction model.
    - criterion: nn.Module, the loss function used during training.
    - optimizer: torch.optim.Optimizer, the optimizer used to update model weights.
    - scheduler: torch.optim.lr_scheduler, adjusts the learning rate based on validation performance.
    - device: str, the device ('cpu' or 'cuda') on which the model is trained.
    - verbose: bool, controls the verbosity of training output.
    """

    def __init__(self, model, lr=0.001, device='cpu', verbose=False):
        """
        Initializes the ChannelModel with the specified parameters.

        Parameters:
        - input_size: int, the number of input features.
        - hidden_size: int, the number of hidden units in the GRU.
        - output_size: int, the number of output features.
        - num_layers: int, the number of GRU layers.
        - lr: float, the learning rate for the optimizer.
        - device: str, the device to use for training ('cpu' or 'cuda').
        - verbose: bool, whether to print detailed training information.
        """
        self.device = device
        self.model = model.to(self.device)
        self.criterion = NMSELoss()  # Normalized Mean Squared Error loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-4)
        self.verbose = verbose

    def prepare_data(self, x_train, y_train, x_valid, y_valid, batch_size):
        # Prepare data loaders for training and validation
        # Create a dataset instance using the provided data and targets, and move it to the specified device
        dataset_train = TensorDataset(torch.from_numpy(x_train).to(self.device),
                                    torch.from_numpy(y_train).to(self.device))
        x_valid_tensor = torch.from_numpy(x_valid).to(self.device)
        y_valid_tensor = torch.from_numpy(y_valid).to(self.device)
        
        # Create a DataLoader to iterate over the dataset with the specified batch size and shuffling option
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        return train_loader, x_valid_tensor, y_valid_tensor

    def train(self, train_loader, x_valid_tensor, y_valid_tensor, num_epochs=10, patience=30):
        """
        Trains the GRU neural network model.

        Parameters:
        - train_loader: DataLoader, the data loader for the training dataset.
        - val_loader: DataLoader, the data loader for the validation dataset.
        - num_epochs: int, the total number of epochs for training.
        - patience: int, the number of epochs with no improvement after which training will be stopped.

        Returns:
        - training_losses: list of floats, the training loss recorded at each iteration.
        - validation_losses: list of floats, the validation loss recorded at each iteration.
        - elapsed_time: float, the total training time in seconds.
        """
        # initialize training time
        start_time = time.time()
        # Initialize variables for logging loss
        avg_train_losses = []
        training_losses = []
        validation_losses = []

        # Initialize early stopping
        best_val_loss = float('inf')
        epochs_since_improvement = 0

        for epoch in range(num_epochs):
            total_train_loss = 0
            self.model.train()
            for (inputs, targets) in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                # Log loss and accumulate for average loss
                training_losses.append(loss.item())
                total_train_loss += loss.item()
            
            avg_train_losses.append(total_train_loss/len(train_loader))

            # Update learning rate scheduler
            self.scheduler.step(loss.item())

            # Run validation
            avg_val_loss = self.evaluate(x_valid_tensor, y_valid_tensor)
            validation_losses.append(avg_val_loss)
            if self.verbose:
                elapsed_time = time.time() - start_time
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {avg_train_losses[-1]:.4f}, '
                      f'Val. Loss: {avg_val_loss:.4f}, '
                      f'Time Elapsed: {elapsed_time:.2f} s')

            # Update best validation loss for early stopping
            if avg_val_loss < best_val_loss * 0.99:
                best_val_loss = avg_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # Check for early stopping
            if epochs_since_improvement >= patience:
                print("Early stopping triggered")
                break

        elapsed_time = time.time() - start_time
        return avg_train_losses, validation_losses, elapsed_time
    
    def update(self, x_train: np.ndarray, y_train: np.ndarray):
        self.optimizer.zero_grad()
        outputs = self.model(torch.from_numpy(x_train).to(self.device))
        loss = self.criterion(outputs, torch.from_numpy(y_train).to(self.device))
        loss.backward()
        self.optimizer.step()
        
        training_loss = loss.item()

        # Update learning rate scheduler
        self.scheduler.step(loss.item())

        return training_loss


    def evaluate(self, x_valid: torch.Tensor, y_valid: torch.Tensor):
        """
        Evaluates the model's performance on the provided dataset.

        Parameters:
        - data_loader: DataLoader, the data loader for the dataset to be evaluated.

        Returns:
        - average_loss: float, the average loss over all batches in the data_loader.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            outputs = self.model(x_valid)
            total_loss = self.criterion(outputs, y_valid).item()
        return total_loss

    def predict(self, x: np.ndarray):
        """
        Predicts the output using the model for the given input data.

        Parameters:
        - x: np.ndarray, input data for which predictions are to be made.

        Returns:
        - predictions: torch.Tensor, predicted output data.
        """
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Move input data to the appropriate device
        device = self.device
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

        # Disable gradient calculation for inference
        with torch.no_grad():
            # Run the model on the entire dataset
            predictions = self.model(x_tensor)

        return predictions
        
    def save_model_state_dictionary(self, filename, horizon):
        """
        Saves the model's state dictionary and associated metadata to a file.

        Parameters:
        - filename: str, the desired filename for saving the model state. If not ending with '.pth', it will be appended.
        - horizon: int, the prediction horizon, used in metadata for descriptive purposes.

        Returns:
        - new_filename: str, the actual filename used for saving the model state dictionary.
        """
        # Ensure the filename has a .pth extension
        base, _ = os.path.splitext(filename)
        new_filename = base + '.pth'

        # Move the model to CPU for saving
        self.model.to("cpu")

        # Create metadata for the model
        metadata = ChannelPredictionMetadata(
            'gru',  # Model type
            horizon,  # Prediction horizon
            'GRU channel prediction network'  # Description
        )

        # Save the model state dictionary and metadata to the specified file
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'metadata': metadata
        }, new_filename)

        # Return the filename used for saving
        return new_filename
    
    def load_model_state_dictionary(self, filename):
        """
        Loads the model's state dictionary and associated metadata from a file.

        Parameters:
        - filename: str, the path to the file containing the saved model state.

        Returns:
        - self: the updated instance of the class with the loaded model state and metadata.

        Raises:
        - ValueError: if the filename does not have a '.pth' extension.
        - FileNotFoundError: if the specified file does not exist.
        """
        # Check if the filename has a .pth extension
        if not filename.endswith('.pth'):
            raise ValueError("The file must have a .pth extension.")

        # Check if the file exists
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"The file '{filename}' does not exist.")

        # Register metadata and other necessary objects as safe for loading
        torch.serialization.add_safe_globals([ChannelPredictionMetadata])
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        torch.serialization.add_safe_globals([np.dtype])
        torch.serialization.add_safe_globals([np.dtypes.Float32DType])
        
        # Load the checkpoint from the specified file
        checkpoint = torch.load(filename, map_location=torch.device(self.device), weights_only=False)
        metadata = checkpoint['metadata']
        
        # Load the model's state dictionary from the checkpoint
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Call flatten_parameters to optimize the model's internal state
        self.model.flatten_parameters()
        
        # Load the metadata and update the model's horizon
        self.horizon = metadata.horizon
        print(f"Loaded {metadata.description}")
        
        # Return the updated instance
        return self
        
    def save_training_results(self, filename, training_loss, validation_loss, valid_freq, num_train):
        """
        Saves the training and validation results along with model parameters to a .mat file.

        Parameters:
        - filename: str, the desired filename for saving the results. It should end with '.mat'.
        - training_loss: list or array-like, the recorded training loss over epochs or iterations.
        - validation_loss: list or array-like, the recorded validation loss over epochs or iterations.
        - valid_freq: int, the frequency of validation during training.
        - num_train: int, the number of training samples.

        Returns:
        - None
        """
        # Gather scheduler parameters
        scheduler_params = {
            'factor': self.scheduler.factor,
            'patience': self.scheduler.patience,
            'min_lr': self.scheduler.min_lrs,
        }

        # Gather optimizer parameters
        optimizer_params = {
            'learning_rate': self.optimizer.param_groups[0]["lr"], 
            'weight_decay': self.optimizer.param_groups[0]["weight_decay"], 
        }

        # Gather network parameters
        network_params = {
            'hidden_size': self.model.gru.hidden_size, 
            'num_layers': self.model.gru.num_layers,
        }

        # Prepare data dictionary for saving
        data_to_save = {
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'valid_freq': valid_freq,
            'num_train': num_train,
            'network_params': network_params,
            'optimizer_params': optimizer_params,
            'scheduler_params': scheduler_params
        }

        # Save the data to a .mat file using scipy
        scipy.io.savemat(filename, data_to_save)

# Example code
if __name__ == "__main__":
    import scipy
    import numpy as np
    from datetime import datetime
    import torch

    # Define parameters
    horizon = 10
    # Load xTraining, yTraining, xValidation, yValidation, Ntx from
    # a mat file. Replace mat_file_name with a valid file name.
    data_file_name = "mat_file_name.mat"

    # Load data from .mat file
    loaded_data = scipy.io.loadmat(data_file_name)
    
    # Split data into training and validation sets
    num_train = 90000
    num_test = 10000
    x_train = loaded_data["xTraining"][:num_train, :, :]
    y_train = loaded_data["yTraining"][:num_train, :]
    x_valid = loaded_data["xValidation"][:num_test, :, :]
    y_valid = loaded_data["yValidation"][:num_test, :]

    # Determine input and output sizes based on the number of transmit antennas
    num_tx_ant = loaded_data["Ntx"][0][0]
    input_size = 2 * num_tx_ant
    hidden_size = 64
    output_size = 2 * num_tx_ant
    num_layers = 2
    lr = 5e-3
    epochs = 20
    valid_freq = 5
    batch_size = 128

    # Normalize the data
    x_max = np.max(x_train)
    x_min = np.min(x_train)
    x_train = (x_train - x_min) / (x_max - x_min)
    y_train = (y_train - x_min) / (x_max - x_min)
    x_valid = (x_valid - x_min) / (x_max - x_min)
    y_valid = (y_valid - x_min) / (x_max - x_min)

    # Select the device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chanpre_model = ChannelPredictorGRU(input_size, hidden_size, output_size, num_layers)

    # Initialize the channel prediction manager
    chanpre = ChannelPredictorManager(lr=lr, num_layers=num_layers, device=device, verbose=True)

    # Prepare data loaders for training and validation
    # Create DataLoaders
    # Create a dataset instance using the provided data and targets, and move it to the specified device
    dataset_train = TensorDataset(torch.from_numpy(x_train).to(device),
                                  torch.from_numpy(y_train).to(device))
    dataset_val = TensorDataset(torch.from_numpy(x_valid).to(device),
                                  torch.from_numpy(y_valid).to(device))
       
    # Create a DataLoader to iterate over the dataset with the specified batch size and shuffling option
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Train the model
    training_loss, validation_loss, elapsed_time = chanpre.train(train_loader, val_loader, 
                                                                 num_epochs=epochs, 
                                                                 patience=np.inf, plot_progress=True)
    
    # Save training results
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m_%d_%H_%M")
    chanpre.save_training_results(f'{data_file_name}_training_results_{formatted_time}.mat',
                                        training_loss=training_loss, validation_loss=validation_loss,
                                        valid_freq=valid_freq, num_train=num_train)

    # Evaluate the model on validation data
    test_loss = chanpre.evaluate(val_loader)
    print(f'Test Loss: {test_loss}')

    # Predict on validation data
    predictions = chanpre.predict(x_valid)

    # Save the model state
    chanpre.save_model_state_dictionary("temp.pth", horizon)

