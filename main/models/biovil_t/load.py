import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from .pretrained import get_biovil_t_image_encoder
from .transformer import VisionTransformerPooler


class BioVILTWrapper(nn.Module):
    def __init__(self, model_name: str = "microsoft/BiomedVLP-BioViL-T"):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=False,
        )
        self.image_model = get_biovil_t_image_encoder()

        self.temporal_pooler = self._find_temporal_pooler()
        if self.temporal_pooler is None:
            print("Warning: VisionTransformerPooler not found inside image_model.")
        else:
            print("Temporal pooler found:", type(self.temporal_pooler))

    def _find_temporal_pooler(self):
        for module in self.image_model.modules():
            if isinstance(module, VisionTransformerPooler):
                return module
        return None

    # -------------------------
    # TEXT
    # -------------------------
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_emb = self.text_model.get_projected_text_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return F.normalize(text_emb, dim=-1)

    def encode_text_intermediates(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        outputs = self.text_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            output_cls_projected_embedding=True,
            return_dict=True,
        )

        cls_proj = outputs.cls_projected_embedding
        if cls_proj is not None:
            cls_proj = F.normalize(cls_proj, dim=-1)

        last_hidden_state = outputs.hidden_states[-1] if outputs.hidden_states is not None else None

        return {
            "cls_projected_embedding": cls_proj,       # [B, D]
            "hidden_states": outputs.hidden_states,    # tuple[layer] of [B, T, H]
            "attentions": outputs.attentions,          # tuple[layer] of [B, heads, T, T]
            "last_hidden_state": last_hidden_state,    # [B, T, H]
        }

    def project_text_hidden_states(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Project token hidden states into the joint image-text space.

        Args:
            last_hidden_state: [B, T, H]

        Returns:
            [B, T, D]
        """
        token_proj = self.text_model.cls_projection_head(last_hidden_state)
        token_proj = F.normalize(token_proj, dim=-1)
        return token_proj

    def encode_text_tokens_projected(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project every token into the same joint space as image embeddings.

        Returns:
            [B, T, D]
        """
        inter = self.encode_text_intermediates(input_ids, attention_mask)
        last_hidden_state = inter["last_hidden_state"]
        if last_hidden_state is None:
            raise RuntimeError("No text hidden states returned.")

        return self.project_text_hidden_states(last_hidden_state)

    # -------------------------
    # IMAGE
    # -------------------------
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        out = self.image_model(
            current_image=images,
            previous_image=None,
        )
        return F.normalize(out.projected_global_embedding, dim=-1)

    def encode_image_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns projected patch embeddings in the joint space.

        Shape:
            [B, H, W, D]
        """
        return self.image_model.get_patchwise_projected_embeddings(
            current_image=images,
            previous_image=None,
            normalize=True,
        )

    def encode_single_image_intermediates(
        self,
        images: torch.Tensor,
    ) -> dict:
        """
        Returns:
        - single-image global embedding
        - single-image patch features from the pooler
        - projected patch embeddings in joint space
        - attentions from the VisionTransformerPooler blocks
        """
        if self.temporal_pooler is None:
            raise RuntimeError("No VisionTransformerPooler found in image_model.")

        captured = {}

        def pre_hook(module, args, kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        handle = self.temporal_pooler.register_forward_pre_hook(pre_hook, with_kwargs=True)

        try:
            image_out = self.image_model(
                current_image=images,
                previous_image=images,
            )
        finally:
            handle.remove()

        if "args" not in captured:
            raise RuntimeError(
                "VisionTransformerPooler was not hit during single-image forward."
            )

        pooler_args = captured["args"]
        pooler_kwargs = dict(captured["kwargs"])
        pooler_kwargs["return_attentions"] = True

        current_patch_features, attentions = self.temporal_pooler(
            *pooler_args, **pooler_kwargs
        )

        global_emb = None
        if hasattr(image_out, "projected_global_embedding") and image_out.projected_global_embedding is not None:
            global_emb = F.normalize(image_out.projected_global_embedding, dim=-1)

        patch_proj = self.image_model.get_patchwise_projected_embeddings(
            current_image=images,
            previous_image=None,
            normalize=True,
        )

        return {
            "global_embedding": global_emb,          # [B, D]
            "patch_features": current_patch_features, # [B, C, H, W]
            "patch_proj": patch_proj,                 # [B, H, W, D]
            "attentions": attentions,                 # list[layer] of [B, heads, L, L]
        }
    # -------------------------
    # TEMPORAL IMAGE SUPPORT
    # -------------------------
    def _call_image_model_temporal(
        self,
        current_images: torch.Tensor,
        previous_images: torch.Tensor,
    ):
        return self.image_model(
            current_image=current_images,
            previous_image=previous_images,
        )

    def _call_patchwise_temporal(
        self,
        current_images: torch.Tensor,
        previous_images: torch.Tensor,
    ):
        return self.image_model.get_patchwise_projected_embeddings(
            current_image=current_images,
            previous_image=previous_images,
            normalize=True,
        )

    def encode_temporal_image(
        self,
        current_images: torch.Tensor,
        previous_images: torch.Tensor,
    ) -> torch.Tensor:
        out = self._call_image_model_temporal(current_images, previous_images)
        return F.normalize(out.projected_global_embedding, dim=-1)

    def encode_temporal_intermediates(
        self,
        current_images: torch.Tensor,
        previous_images: torch.Tensor,
    ) -> dict:
        """
        Returns:
          - temporal global embedding from temporal forward path
          - fused patch features from the temporal image model output
          - projected fused patch embeddings in joint space
          - temporal attentions from the modified VisionTransformerPooler
        """
        if self.temporal_pooler is None:
            raise RuntimeError("No VisionTransformerPooler found in image_model.")

        captured = {}

        def pre_hook(module, args, kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        handle = self.temporal_pooler.register_forward_pre_hook(pre_hook, with_kwargs=True)

        try:
            image_out = self._call_image_model_temporal(current_images, previous_images)
        finally:
            handle.remove()

        if "args" not in captured:
            raise RuntimeError(
                "Temporal pooler was not hit during forward. "
                "This usually means the loaded image_model is not using your modified local pooler."
            )

        pooler_args = captured["args"]
        pooler_kwargs = dict(captured["kwargs"])
        pooler_kwargs["return_attentions"] = True

        current_patch_features, temporal_attentions = self.temporal_pooler(
            *pooler_args, **pooler_kwargs
        )

        temporal_global = None
        if hasattr(image_out, "projected_global_embedding") and image_out.projected_global_embedding is not None:
            temporal_global = F.normalize(image_out.projected_global_embedding, dim=-1)

        temporal_patch_proj = self._call_patchwise_temporal(current_images, previous_images)

        return {
            "temporal_global_embedding": temporal_global,     # [B, D] or None
            "current_patch_features": current_patch_features, # [B, C, H, W]
            "temporal_patch_proj": temporal_patch_proj,       # [B, H, W, D]
            "temporal_attentions": temporal_attentions,       # list[layer] of [B, heads, 2L, 2L]
            "image_model_output": image_out,                  # raw ImageModelOutput
        }

    # -------------------------
    # FORWARD
    # -------------------------
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        outputs = {}

        if pixel_values is not None:
            outputs["image_emb"] = self.encode_image(pixel_values)

        if input_ids is not None and attention_mask is not None:
            outputs["text_emb"] = self.encode_text(input_ids, attention_mask)

        return outputs


def load_biovil_t(
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    device: str = "cpu",
):
    model = BioVILTWrapper(model_name=model_name)
    model = model.to(device)
    model.eval()
    return model