# gemma3_wrapper/gemma3_vision_components.py

import mlx.core as mx
import mlx.nn as nn

# In a real implementation, this class would be a full, detailed replica
# of the SigLIP architecture found in mlx-vlm. For this PoC, we'll keep
# it as a placeholder that can correctly load the weights.
class VisionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # Placeholder for the full architecture. This would normally contain
        # self.embeddings, self.encoder, self.post_layernorm, etc.
        self.dummy_layer = nn.Linear(1, 1) # To ensure it's a valid nn.Module

    def __call__(self, pixel_values):
        # In a real implementation, this would pass pixel_values through the full ViT.
        # For the PoC, we'll return correctly shaped dummy data.
        batch_size = pixel_values.shape[0]
        # From config.json: vision_config.hidden_size = 1152
        # A vision tower outputs a sequence of patch embeddings.
        # Let's assume a standard number of patches, e.g., 4096 (64x64 grid).
        num_patches = 4096
        hidden_size = 1152
        return mx.random.normal(shape=(batch_size, num_patches, hidden_size))

# This class is a direct MLX translation of Gemma 3's specific projector.
class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, vision_config: dict, text_config: dict):
        super().__init__()
        # From config.json:
        vision_hidden_size = vision_config["hidden_size"]      # 1152
        text_hidden_size = text_config["hidden_size"]          # 5376
        image_size = vision_config["image_size"]               # 896
        patch_size = vision_config["patch_size"]               # 14
        mm_tokens_per_image = text_config["mm_tokens_per_image"] # 256

        # This corresponds to the 'mm_input_projection_weight' in the weight tree
        self.mm_input_projection = nn.Linear(vision_hidden_size, text_hidden_size, bias=False)

        # This corresponds to 'mm_soft_emb_norm'
        self.mm_soft_emb_norm = nn.RMSNorm(vision_hidden_size, eps=vision_config["layer_norm_eps"])

        patches_per_image = image_size // patch_size
        tokens_per_side = int(mm_tokens_per_image**0.5)
        kernel_size = patches_per_image // tokens_per_side

        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        # This logic directly mimics the fusion process seen in other implementations
        b, num_patches, hidden_dim = x.shape
        patches_per_side = int(num_patches**0.5)

        # Reshape for pooling
        x = x.reshape(b, patches_per_side, patches_per_side, hidden_dim)

        pooled_vision_outputs = self.avg_pool(x)

        flat_pooled = pooled_vision_outputs.reshape(b, -1, hidden_dim)

        normed_vision_outputs = self.mm_soft_emb_norm(flat_pooled)
        projected_vision_outputs = self.mm_input_projection(normed_vision_outputs)

        return projected_vision_outputs

def load_vlm_weights(model_path: Path) -> dict:
    all_weights = {}
    for wf in model_path.glob("*.safetensors"):
        all_weights.update(mx.load(str(wf)))

    vlm_weights = {"vision_tower": [], "projector": []}
    for key, value in all_weights.items():
        if key.startswith("vision_tower."):
            new_key = key.replace("vision_tower.", "")
            vlm_weights["vision_tower"].append((new_key, value))
        elif key.startswith("multi_modal_projector."):
            new_key = key.replace("multi_modal_projector.", "")
            vlm_weights["projector"].append((new_key, value))

    return vlm_weights
