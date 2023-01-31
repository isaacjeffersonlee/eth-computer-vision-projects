import os
import torch
import numpy as np
from copy import deepcopy
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import generate_resnet

def evaluate_model(model:nn.Module, loader:DataLoader, 
                   device:str="cpu", mapping=None, 
                   classwise=False) -> float:
    """
    Evaluates the model accuracy on the given dataset. 
    Inputs:
        model: The model to evaluate [nn.Module]
        loader: The dataset loader [DataLoader]
        device: The device to use for evaluation [str]
        mapping: The mapping of predictions to dataset [list]
        classwise: If True, returns dictionary of accuracies per class [bool]
    """

    if mapping is not None:
        mapping = np.array(mapping)

    # Set the device for the model
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    # Keep track with lists
    preds_list, gt_list = list(), list()

    # Iterate over the dataset
    with torch.no_grad():

        for images, labels in loader:

            # Move the images and labels to the device
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get the predictions
            _, preds = torch.max(outputs, 1)
            preds = preds.numpy()
            if mapping is not None:
                preds = mapping[preds]

            preds_list.extend(preds.tolist())
            gt_list.extend(labels.data.numpy())

    preds = np.array(preds_list, dtype=int)
    gts = np.array(gt_list, dtype=int)

    if not classwise:
        # Return the accuracy
        return np.mean(preds==gts)
    else:
        # Return the class-wise accuracies as dict
        classes = np.unique(gts)
        accs = dict()
        for class_ in classes: 
            bool_class = gts == class_
            preds_class = preds[bool_class]
            accs[class_] = float(np.mean(preds_class == class_))
        return accs

def check_relu_layer(model, modelr):

    input = torch.randn(10, 10)

    output = model(input)
    outputr = modelr(input)
    outputz = F.relu(output)

    assert torch.allclose(outputr, outputz), "Imlementation is not correct."

    print("Correct implementation!")

def check_freezing(model):
    
    model_copy = deepcopy(model)

    input = torch.randn(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    loss = (model(input)**2).sum()
    loss.backward()
    optimizer.step()

    if not torch.allclose(model.lin2.weight, model_copy.lin2.weight):
        print("Backpropagation is working correctly for lin2 :)")
    else:
        assert False, "Backpropagation is not working correctly for lin2 :("

    if torch.allclose(model.lin1.weight, model_copy.lin1.weight):
        print("lin1 is frozen :)")
    else:
        assert False, "lin1 is not frozen :("

    if torch.allclose(model.lin3.weight, model_copy.lin3.weight):
        print("lin3 is frozen :)")
    else:
        assert False, "lin3 is not frozen :("

    print("Backpropagation is working correctly for all layers! :D")