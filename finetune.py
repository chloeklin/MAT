import os
import os.path as osp
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
os.chdir('src')
from featurization.data_utils import load_data_from_df, construct_loader
from transformer import make_model
from tqdm import tqdm  # Progress bar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='u0')
    args = parser.parse_args()

    qm9_target_to_idx = {
        'mu': 6, 'alpha': 7, 'homo': 8, 'lumo': 9, 'gap': 10, 'r2': 11,
        'zpve': 12, 'cv': 17,
        'u0_atom': 18, 'u298_atom': 19, 'h298_atom': 20, 'g298_atom': 21
    }


    batch_size = 32

    # Formal charges are one-hot encoded to keep compatibility with the pre-trained weights.
    # If you do not plan to use the pre-trained weights, we recommend to set one_hot_formal_charge to False.
    # Define dataset paths
    datasets = {
        "train": "../data/qm9/qm9_train.csv",
        "val": "../data/qm9/qm9_val.csv",
        "test": "../data/qm9/qm9_test.csv"
    }

    # Load data and construct DataLoaders
    data_loaders = {}

    for split, path in datasets.items():
        X, y = load_data_from_df(path, qm9_target_to_idx[args.target], one_hot_formal_charge=True)
        if split == 'test':
            data_loaders[split] = construct_loader(X, y, batch_size=64, shuffle=False)
        else:
            data_loaders[split] = construct_loader(X, y, batch_size)

    # Access loaders
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    test_loader = data_loaders["test"]

    d_atom = X[0][0].shape[1]  # It depends on the used featurization.

    model_params = {
        'd_atom': d_atom,
        'd_model': 1024,
        'N': 8,
        'h': 16,
        'N_dense': 1,
        'lambda_attention': 0.33, 
        'lambda_distance': 0.33,
        'leaky_relu_slope': 0.1, 
        'dense_output_nonlinearity': 'relu', 
        'distance_matrix_kernel': 'exp', 
        'dropout': 0.0,
        'aggregation_type': 'mean'
    }

    model = make_model(**model_params)
    pretrained_name = '../pretrained_weights.pt'  # This file should be downloaded first (See README.md).
    pretrained_state_dict = torch.load(pretrained_name, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model_state_dict = model.state_dict()
    for name, param in pretrained_state_dict.items():
        if 'generator' in name:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        model_state_dict[name].copy_(param)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    loss_fn = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)  # Fine-tuning learning rate

    # Training and evaluation loop
    num_epochs = 100  # Adjust as needed

    # Early stopping parameters
    patience = 10  # Stop training if val loss doesn't improve after 10 epochs
    best_val_loss = float("inf")
    epochs_no_improve = 0

    def train(model, loader, optimizer, loss_fn):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc="Training", leave=False):
            adjacency_matrix, node_features, distance_matrix, y = batch
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

            node_features = node_features.to(device)
            adjacency_matrix = adjacency_matrix.to(device)
            distance_matrix = distance_matrix.to(device)
            batch_mask = batch_mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)

            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(loader)

    def evaluate(model, loader, loss_fn):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                adjacency_matrix, node_features, distance_matrix, y = batch
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

                node_features = node_features.to(device)
                adjacency_matrix = adjacency_matrix.to(device)
                distance_matrix = distance_matrix.to(device)
                batch_mask = batch_mask.to(device)
                y = y.to(device)

                output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
                loss = loss_fn(output, y)
                total_loss += loss.item()

        return total_loss / len(loader)


    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}:")
        
        train_loss = train(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_finetuned_model.pt")  # Save best model
            print("Best model saved!")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs")

        # Stop training if no improvement after patience epochs
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break


    # Evaluate on test set
    print("\nEvaluating on Test Set...")
    test_loss = evaluate(model, test_loader, loss_fn)
    print(f"Test MAE: {test_loss:.6f}")

    # Define the target names
    qm9_target_to_col = {
        'mu': 'mu', 'alpha': 'alpha', 'homo': "HOMO", 'lumo': "LUMO", 'gap': 'gap', 'r2': "R2",
        'zpve': "ZPVE", 'u0_atom': "U0", 'u298_atom': "U", 'h298_atom': "H", 'g298_atom': "G", 'cv': 'Cv',
    }
    target_name = qm9_target_to_col[args.target]  # Get the corresponding target name

    # Define the file name
    csv_filename = "qm9_MAT.csv"

    # Check if the file exists
    if osp.exists(csv_filename):
        # Load the existing CSV
        df = pd.read_csv(csv_filename)
    else:
        # Create a new DataFrame with the specified columns
        df = pd.DataFrame(columns=["mu", "alpha", "HOMO", "LUMO", "R2", "ZPVE", "U0", "U", "H", "G", "Cv", "gap", 
                                "Model", "Mean_Std_MAE", "Mean_Std_logMAE"])

    # Update the test loss for the corresponding target
    df.loc[0, target_name] = test_loss  # Store the test loss in the correct column

    # Ensure the "Model" column stores the model name
    df.loc[0, "Model"] = args.model

    # Save the updated DataFrame back to CSV
    df.to_csv(csv_filename, index=False)


# Example Usage
if __name__ == "__main__":
    main()