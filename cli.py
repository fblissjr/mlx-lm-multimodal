import sys
sys.path.append('.') # <--- THIS IS THE FIX

# Now, all your other imports will work
import argparse
from PIL import Image
from rich.console import Console

# This import will now succeed because '.' (the current directory) is in the path
import gemma3_wrapper
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

# Use rich for pretty printing
console = Console()

def generate_vlm(model: Gemma3VLM, prompt: str, image_path: str, **kwargs):
    """
    The main user-facing VLM generation function.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    # Gemma 3 uses a very specific prompt structure
    formatted_prompt = f"<bos><start_of_turn>user\n<image>{prompt}<end_of_turn>\n<start_of_turn>model\n"

    # 1. FUSE: Get fused embeddings and the special attention mask
    fused_embeddings, attention_mask = model.fuse_inputs(formatted_prompt, image)

    # 2. DELEGATE: Call mlx-lm's generator with the fused inputs
    sampler = make_sampler(temp=kwargs.get("temp", 0.0))
    token_stream = stream_generate(
        model=model.language_model,
        tokenizer=model.processor.tokenizer,
        prompt=[], # CRITICAL: Pass an empty prompt list
        input_embeddings=fused_embeddings,
        mask=attention_mask, # Pass the custom mask
        max_tokens=kwargs.get("max_tokens", 100),
        sampler=sampler
    )

    # 3. YIELD: Stream the results back to the user.
    response = ""
    print("\nAssistant: ", end="", flush=True)
    for chunk in token_stream:
        print(chunk.text, end="", flush=True)
        response += chunk.text
    print("\n")
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 3 VLM Wrapper PoC")
    parser.add_argument("--model", type=str, required=True, help="Path to the local model directory.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature.")

    args = parser.parse_args()

    console.print(f"[bold green]Loading model from {args.model}...[/bold green]")
    vlm_model = load_gemma3_vlm(args.model)
    console.print("[bold green]✓ Model loaded.[/bold green]")

    generate_vlm(
        vlm_model,
        args.prompt,
        args.image,
        max_tokens=args.max_tokens,
        temp=args.temp,
    )
