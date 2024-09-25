import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from modelscopy2 import ResNet

class ResNetWithGradCAM(ResNet):
    
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, inputs):
        out = self.stg1(inputs)
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + self.convShortcut3(out))
        out = F.relu(self.ident3(out) + out)
        
        #stage4             
        out = F.relu(self.conv4(out) + self.convShortcut4(out))
        out = F.relu(self.ident4(out) + out)
        
        # Here we extract the activations for the last convolutional block
        self.activations = out
        h = out.register_hook(self.activations_hook)  # Register hook to capture gradients
        
        # Pooling and classification
        out = self.avgpool(out)
        out = self.classifier(out)
        
        return out
    
    def get_heatmap(self, class_idx):
        """
        Compute the Grad-CAM heatmap for a specific class index.
        """
        # Compute the gradient of the output with respect to the last convolutional layer
        self.zero_grad()
        class_score = self.activations[:, class_idx].sum()
        class_score.backward(retain_graph=True)

        # Get the gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Compute the weight for each channel in the activation maps
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        
        # Compute the weighted sum of the activations
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU to keep positive values only
        
        # Normalize the CAM to [0, 1] range
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam

