import torch
import torch.nn.functional as F
import torch.nn as nn

from pipelines.biovil_t_pipeline import (
    get_device, 
    load_biovil_t, 
    build_single_loader, 
    SingleInferenceDataset,
    load_biovil_preprocessor,
)


def find_target_layer(
    module: nn.Module,
    layer_name: str | None = None,
):
    """
    Find a target layer for Grad-CAM.

    If layer_name is provided:
        Returns the module with that name.

    Otherwise:
        Returns the last Conv2d layer in the model.
    """

    modules = dict(module.named_modules())

    if layer_name is not None:
        if layer_name not in modules:
            available = "\n".join(modules.keys())
            raise ValueError(
                f"Layer '{layer_name}' not found.\n"
                f"Available layers:\n{available}"
            )

        target = modules[layer_name]
        print(f"Using user-specified Grad-CAM layer: {layer_name} ({type(target)})")
        return target

    last_name = None
    last_module = None

    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Conv2d):
            last_name = name
            last_module = submodule

    if last_module is None:
        raise RuntimeError("No Conv2d layer found in image model.")

    print(f"Using last Conv2d layer for Grad-CAM: {last_name} ({type(last_module)})")
    return last_module



class BioViLTGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = None
        self.backward_handle = None

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def register_hooks(self):
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def remove_hooks(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()
        if self.backward_handle is not None:
            self.backward_handle.remove()

    def __call__(self, pixel_values, input_ids, attention_mask):
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        self.register_hooks()
        self.model.eval()

        try:
            # image forward through BioViL-T image path
            image_out = self.model.image_model(
                current_image=pixel_values,
                previous_image=None,
            )
            image_emb = F.normalize(image_out.projected_global_embedding, dim=-1)

            # report-conditioned text embedding
            text_emb = self.model.encode_text(input_ids, attention_mask)

            # target scalar: image-text similarity
            score = torch.sum(image_emb * text_emb, dim=1).sum()
            score.backward()

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

            # Grad-CAM weights: global average pooling over spatial dims
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [B,C,1,1]

            cam = (weights * self.activations).sum(dim=1, keepdim=True)   # [B,1,H,W]
            cam = F.relu(cam)

            cam = F.interpolate(
                cam,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            cam = cam.squeeze(1)   # [B,H,W]
            cam = cam - cam.amin(dim=(1, 2), keepdim=True)
            cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)

            return cam.detach().cpu(), score.item()

        finally:
            self.remove_hooks()


class BioViLTGradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = None
        self.backward_handle = None

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        self.activations.retain_grad()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def register_hooks(self):
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def remove_hooks(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def __call__(self, pixel_values, input_ids, attention_mask):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        self.register_hooks()

        try:
            # image forward
            image_out = self.model.image_model(
                current_image=pixel_values,
                previous_image=None,
            )
            image_emb = F.normalize(image_out.projected_global_embedding, dim=-1)

            # text forward
            text_emb = self.model.encode_text(input_ids, attention_mask)

            # similarity score
            score = torch.sum(image_emb * text_emb, dim=1).sum()
            score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM++ hooks did not capture activations/gradients.")

            activations = self.activations
            gradients = self.gradients

            # Grad-CAM++ weights
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)

            # sum over spatial dims
            spatial_sum = grad_3.sum(dim=(2, 3), keepdim=True)

            eps = 1e-8
            alpha_num = grad_2
            alpha_denom = 2.0 * grad_2 + activations * spatial_sum
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * eps)

            alphas = alpha_num / (alpha_denom + eps)

            positive_gradients = F.relu(gradients)
            weights = (alphas * positive_gradients).sum(dim=(2, 3), keepdim=True)   # [B,C,1,1]

            cam = (weights * activations).sum(dim=1, keepdim=True)   # [B,1,H,W]
            cam = F.relu(cam)

            cam = F.interpolate(
                cam,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            cam = cam.squeeze(1)   # [B,H,W]
            cam = cam - cam.amin(dim=(1, 2), keepdim=True)
            cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)

            return cam.detach().cpu(), score.item()

        finally:
            self.remove_hooks()




def init_grad_cam(
    single_csv_path: str,
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    device: str | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    target_layer_name: str | None = None,
    target_layer: nn.Module | None = None,
):
    if device is None:
        device = get_device()

    # preprocessors
    image_processor = load_biovil_preprocessor(model_name=model_name, mode="image")
    tokenizer = load_biovil_preprocessor(model_name=model_name, mode="text")

    # dataloader
    single_loader = build_single_loader(
        single_ds=SingleInferenceDataset(single_csv_path),
        image_processor=image_processor,
        tokenizer=tokenizer,
        single_batch_size=batch_size,
        num_workers=num_workers,
    )

    batch = next(iter(single_loader))

    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # model
    model = load_biovil_t(model_name=model_name, device=device)
    model.eval()

    # choose target layer
    if target_layer is None:
        target_layer = find_target_layer(
            model.image_model,
            layer_name=target_layer_name,
        )

    # grad-cam
    grad_cam = BioViLTGradCAM(model=model, target_layer=target_layer)
    cam, score = grad_cam(pixel_values, input_ids, attention_mask)

    return cam, score, batch, model, target_layer



def init_grad_cam_pp(
    single_csv_path: str,
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    device: str | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    target_layer_name: str | None = None,
    target_layer: nn.Module | None = None,
):
    if device is None:
        device = get_device()

    # preprocessors
    image_processor = load_biovil_preprocessor(model_name=model_name, mode="image")
    tokenizer = load_biovil_preprocessor(model_name=model_name, mode="text")

    # dataloader
    single_loader = build_single_loader(
        single_ds=SingleInferenceDataset(single_csv_path),
        image_processor=image_processor,
        tokenizer=tokenizer,
        single_batch_size=batch_size,
        num_workers=num_workers,
    )

    batch = next(iter(single_loader))

    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # model
    model = load_biovil_t(model_name=model_name, device=device)
    model.eval()

    # choose target layer
    if target_layer is None:
        target_layer = find_target_layer(
            model.image_model,
            layer_name=target_layer_name,
        )

    # Grad-CAM++
    grad_cam_pp = BioViLTGradCAMPlusPlus(
        model=model,
        target_layer=target_layer,
    )
    cam, score = grad_cam_pp(pixel_values, input_ids, attention_mask)

    return cam, score, batch, model, target_layer