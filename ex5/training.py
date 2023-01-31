
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

def train_model(model:nn.Module, train_loader:DataLoader, val_loader:DataLoader, 
                num_epochs:int=10, criterion=None, optimizer=None,
                best_of:str="loss", device:str="cpu", 
                save_path:str=f"./ckpt/model.pt"):
    """
    Trains the model on the given dataset. Selects the best model based on the
    validation set and saves it to the given path. 
    Inputs: 
        model: The model to train [nn.Module]
        train_loader: The training data loader [DataLoader]
        val_loader: The validation data loader [DataLoader]
        num_epochs: The number of epochs to train for [int]
        criterion: The loss function [Any]
        optimizer: The optimizer [Any]
        best_of: The metric to use for validation [str: "loss" or "accuracy"]
        device: The device to train on [str: cpu, cuda, or mps]
        save_path: The path to save the model to [str]
    Output:
        Dictionary containing the training and validation losses and accuracies
        at each epoch. Also contains the epoch number of the best model.
    """

    # Check that the best_of parameter is valid
    best_of = best_of.lower()
    assert best_of in ["loss", "accuracy"], "best_of must be 'loss' or 'accuracy'"
    print("Validation metric:", best_of)

    # Set the best validation metric to \pm infinity
    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    best_val_epoch = 0

    # Logs for the training
    log_train_acc, log_train_loss = list(), list()
    log_val_acc, log_val_loss = list(), list()

    # Set the device for the model
    model.to(device)
    print(f"Training on device: {device}")

    pbar = tqdm(range(num_epochs), desc="Epochs")

    for epoch in pbar:

        ###############################
        # Train the model for one epoch
        ###############################

        # Keep track of the number of correct predictions and loss
        all_corrects, all_samples = 0, 0
        total_loss = 0.0

        # Set the model to training mode
        model.train()
        
        for images, labels in train_loader:

            # Move the images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get the predictions
            _, preds = torch.max(outputs, 1)

            # Store the number of correct predictions
            corrects = torch.sum(preds == labels.data)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update loss
            total_loss += loss.item()*len(labels)

            # Update the weights
            optimizer.step()

            # Update the number of correct predictions and samples
            all_corrects += corrects
            all_samples += len(labels)

        # Compute the training accuracy for the epoch
        train_acc = float(all_corrects / all_samples)

        # Compute the training loss for the epoch
        train_loss = total_loss / all_samples

        # Log the training accuracy and loss
        log_train_acc.append(train_acc)
        log_train_loss.append(train_loss)

        ##################################
        # Validate the model for one epoch
        ##################################
        
        # Keep track of the number of correct predictions and loss
        all_corrects, all_samples = 0, 0
        total_loss = 0.0

        # Set the model to evaluation mode
        model.eval()

        with torch.no_grad():
        
            for images, labels in val_loader:

                # Move the images and labels to the device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Get the predictions
                _, preds = torch.max(outputs, 1)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Update loss
                total_loss += loss.item()*len(labels)

                # Store the number of correct predictions
                corrects = torch.sum(preds == labels.data)

                # Update the number of correct predictions and samples
                all_corrects += corrects
                all_samples += len(labels)

        # Compute the training accuracy for the epoch
        val_acc = float(all_corrects / all_samples)

        # Compute the training loss for the epoch
        val_loss = total_loss / all_samples

        # Log the training accuracy and loss
        log_val_acc.append(val_acc)
        log_val_loss.append(val_loss)

        # Update the best model

        if best_of == "loss":
            # Save the model if the validation loss has decreased
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
                torch.save(model.state_dict(), save_path)
        else:
            # Save the model if the validation accuracy has increased
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                torch.save(model.state_dict(), save_path)

        # Update the progress bar
        pbar.set_description(f"TR L|A: {train_loss:.4f}|{train_acc:.4f}," 
                           + f" VL L|A: {val_loss:.4f}|{val_acc:.4f}"
                           + f" Best Epoch: {best_val_epoch}")

    training_log = {
        "train_acc": log_train_acc,
        "train_loss": log_train_loss,
        "val_acc": log_val_acc,
        "val_loss": log_val_loss,
        "best_val_epoch": best_val_epoch,
    }

    return training_log
