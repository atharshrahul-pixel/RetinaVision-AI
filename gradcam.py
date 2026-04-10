import torch
import torch.nn.functional as F
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):

        self.model.zero_grad()

        output = self.model(input_tensor)
        loss = output[:, class_idx]

        loss.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1)

        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        return cam


def overlay_heatmap(original_image, heatmap, alpha=0.4):

    # Resize heatmap to original image size
    heatmap = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Ensure original image is uint8
    if original_image.dtype != np.uint8:
        original_image = np.uint8(original_image)

    overlayed = cv2.addWeighted(
        original_image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlayed