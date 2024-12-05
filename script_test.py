from PIL import Image
import os, sys
import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
import argparse
import numpy as np
 
parser = argparse.ArgumentParser(description="Python Script Traffic Sign Classification",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--image", default='test.jpg', type=str, help="image to classify")
parser.add_argument("-m", "--model", default='CNN_4Conv_1Linear_1024.pth', type=str, help="model saved file")
config = vars(parser.parse_args())
print(f"Classify {config['image']} using saved model {config['model']}")

# Check save dir
SAVE_DIR = "DeepLearningPytorch_INFO6147/Project/"
if not os.path.isdir(SAVE_DIR):
    SAVE_DIR=''# Current dir -> Change this
    
class CNN_Model(nn.Module):
    def __init__(self, num_classes=10, 
                 conv_channels=[32, 64, 128, 256], 
                 use_batch_normalization=False,
                 conv_dropouts=[0.3, 0.3, 0.3, 0.3],
                 fc_units=1024
                 ):
        super(CNN_Model, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        
        in_channels = 3  # Input image has 3 channels (RGB)
        
        for out_channels in conv_channels:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
            if use_batch_normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            if conv_dropouts != None:
                layers.append(nn.Dropout(p=0.3))
            self.conv_layers.append(nn.Sequential(*layers))
            in_channels = out_channels

        # Calculate the size after the last conv block to define the input size for the fully connected layer
        self.fc_input_size = conv_channels[-1] * (100 // (2 ** len(conv_channels))) ** 2
        self.block_fc = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            # Fully Connected Layers
            nn.Linear(self.fc_input_size, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, num_classes)
        )
        
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.block_fc(x)
        return x

def test_real_image(imgs_path, model_path, class_name, IMG_SIZE, device):
    # Load model (not just state dict)
    model = torch.load(model_path, weights_only=False)
    model.eval().to(device)

    # Transform for real images
    transform_test_data = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for img_path in imgs_path:
        # Load the image using PIL 
        pil_image = Image.open(img_path).convert('RGB') 
        pil_image = transform_test_data(np.array(pil_image).astype('float'))
        # Apply the transformations 
        input_tensor = pil_image.unsqueeze(0).to(device) # Add batch dimension

        with torch.no_grad(): 
            output = model(input_tensor).cpu()
            _, predicted = torch.max(output, 1)
            prob = list(output[0])[predicted.item()].item()
            # Print the predicted class 
            print (f"Output prediction: {output}")
            print(f"Predicted Class: {class_name[predicted.item()]} with id {predicted.item()}, raw output {prob}")

            # Display the image and the predicted label using matplotlib 
            plt.imshow(pil_image)
            plt.title(f"Predicted Class: {class_name[predicted.item()]}") 
            plt.axis('off') # Turn off axis labels
            plt.show()

class_dict_new={0: 'Speed limit (40km/h)',
                1: 'Speed limit (60km/h)',
                2: 'speed limit (80km/h)',
                3: 'Dont Go Left',
                4: 'No Car',
                5: 'watch out for cars',
                6: 'Bicycles crossing',
                7: 'Zebra Crossing',
                8: 'No stopping',
                9: 'No entry'}

test_real_image(imgs_path=[config['image']],
                model_path=SAVE_DIR+config['model'],
                class_name=class_dict_new,
                device='cpu',
                IMG_SIZE=100)