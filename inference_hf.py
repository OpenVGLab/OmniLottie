"""
OmniLottie Inference Script - Hugging Face Compatible Version

Uses decoder_hf.py with from_pretrained() to load models.
Supports automatic model downloading from Hugging Face Hub.

Usage:
    # Text-to-Lottie (from HF Hub)
    python inference_hf.py --model_path OmniLottie/OmniLottie --text "A bouncing ball"

    # Video-to-Lottie (local model)
    python inference_hf.py --model_path ./model --video video.mp4

    # Image-to-Lottie
    python inference_hf.py --model_path OmniLottie/OmniLottie --image image.png --text "rotating animation"
"""

import os
import torch
import argparse
import json
import re
from pathlib import Path
from PIL import Image
import numpy as np
from decord import VideoReader, cpu

# Import HF-compatible model
from decoder_hf import LottieDecoder
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# Import Lottie conversion tools
from lottie.objects.lottie_tokenize import LottieTensor
from lottie.objects.lottie_param import (
    from_sequence, ShapeLayer, NullLayer, PreCompLayer, TextLayer,
    SolidColorLayer, Font, Chars,
    shape_layer_to_json, null_layer_to_json, precomp_layer_to_json,
    text_layer_to_json, solid_layer_to_json, font_to_json, char_to_json
)

# Constants
SYSTEM_PROMPT = "You are a Lottie animation expert."
VIDEO_PROMPT = "Turn this video into Lottie code."
LOTTIE_BOS = 192398
LOTTIE_EOS = 192399
PAD_TOKEN = 151643


def simplify_to_animation_description(text):
    """Simplify text prompt to animation description"""
    if not text or not isinstance(text, str):
        return text

    prefixes = [
        r'^The video features?\s+', r'^The scene shows?\s+',
        r'^An animation of\s+', r'^There is\s+', r'^It shows?\s+'
    ]

    for pattern in prefixes:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    if text:
        text = text[0].upper() + text[1:]

    return text.strip()


def load_frames_from_video(video_path, num_frames=8, max_size=336):
    """Load frames from video file (matches app_hf.py)"""
    ext = os.path.splitext(video_path)[1].lower()
    frames = []

    if ext in ('.gif', '.webp'):
        try:
            img = Image.open(video_path)
            total_frames = getattr(img, 'n_frames', 1)

            if total_frames < 1:
                raise ValueError(f"No frames in {ext.upper()}: {video_path}")

            indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames)).astype(int)

            for idx in indices:
                img.seek(idx)
                frame = img.convert('RGB')
                if max(frame.size) > max_size:
                    frame.thumbnail((max_size, max_size), Image.LANCZOS)
                frames.append(frame)

            img.close()

        except Exception as e:
            raise ValueError(f"Failed to load {ext.upper()}: {str(e)}")

    else:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            if total_frames < 1:
                raise ValueError(f"Video has no frames: {video_path}")

            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            frames_np = vr.get_batch(indices).asnumpy()

            for f in frames_np:
                img = Image.fromarray(f)
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.LANCZOS)
                frames.append(img)

        except Exception as e:
            raise ValueError(f"Failed to load video: {str(e)}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames


def build_messages(task_type, text_prompt=None, image=None, video_frames=None):
    """Build messages for inference (matches app_hf.py)"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if task_type == "text":
        text = simplify_to_animation_description(text_prompt)
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"Generate Lottie code: {text}"}]
        })

    elif task_type == "image":
        text = simplify_to_animation_description(text_prompt) if text_prompt else "A simple animation"
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Animate this image: {text}"}
            ]
        })

    elif task_type == "video":
        messages.append({
            "role": "user",
            "content": [
                {"type": "video", "video": video_frames, "fps": 8.0},
                {"type": "text", "text": VIDEO_PROMPT}
            ]
        })

    return messages


def prepare_inference_input(processor, messages, device):
    """Prepare input for inference (matches app_hf.py exactly)"""
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=False,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    target_len = 1500

    if input_ids.shape[1] < target_len:
        pad_len = target_len - input_ids.shape[1]
        input_ids = torch.cat([
            torch.full((1, pad_len), PAD_TOKEN, dtype=torch.long),
            input_ids
        ], dim=1)
        attention_mask = torch.cat([
            torch.zeros((1, pad_len), dtype=torch.long),
            attention_mask
        ], dim=1)

    result = {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'pixel_values': inputs.get('pixel_values').to(device) if inputs.get('pixel_values') is not None else None,
        'image_grid_thw': inputs.get('image_grid_thw').to(device) if inputs.get('image_grid_thw') is not None else None,
        'pixel_values_videos': inputs.get('pixel_values_videos').to(device) if inputs.get('pixel_values_videos') is not None else None,
        'video_grid_thw': inputs.get('video_grid_thw').to(device) if inputs.get('video_grid_thw') is not None else None,
    }

    return result


def generate_lottie(model, inputs, max_tokens, device, use_sampling=False, temperature=0.9, top_p=0.25, top_k=5):
    """Generate Lottie tokens (matches app_hf.py exactly)"""
    model.transformer.rope_deltas = None
    position_ids, _ = model.transformer.get_rope_index(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        image_grid_thw=inputs.get('image_grid_thw'),
        video_grid_thw=inputs.get('video_grid_thw'),
    )
    position_ids = position_ids * inputs['attention_mask'][None, ]

    kwargs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'pixel_values': inputs.get('pixel_values'),
        'image_grid_thw': inputs.get('image_grid_thw'),
        'pixel_values_videos': inputs.get('pixel_values_videos'),
        'video_grid_thw': inputs.get('video_grid_thw'),
        'position_ids': position_ids,
        'max_new_tokens': max_tokens,
        'eos_token_id': LOTTIE_EOS,
        'pad_token_id': PAD_TOKEN,
        'use_cache': True,
    }

    if use_sampling:
        kwargs.update({'do_sample': True, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k})
    else:
        kwargs.update({'do_sample': False, 'num_beams': 1})

    with torch.no_grad():
        outputs = model.transformer.generate(**kwargs)

    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_len:].tolist()

    del outputs, kwargs, position_ids

    if generated_ids and generated_ids[0] == LOTTIE_BOS:
        generated_ids = generated_ids[1:]
    if LOTTIE_EOS in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(LOTTIE_EOS)]

    return generated_ids


def tokens_to_lottie_json(generated_ids):
    """Convert generated tokens to Lottie JSON format"""
    reconstructed_tensor = LottieTensor.from_list(generated_ids)
    reconstructed_sequence = reconstructed_tensor.to_sequence()
    reconstructed = from_sequence(reconstructed_sequence)

    json_animation = {
        "v": reconstructed.get("v", "5.5.2"),
        "fr": reconstructed.get("fr", 8),
        "ip": reconstructed.get("ip", 0),
        "op": reconstructed.get("op", 16),
        "w": reconstructed.get("w", 512),
        "h": reconstructed.get("h", 512),
        "nm": reconstructed.get("nm", "Animation"),
        "ddd": reconstructed.get("ddd", 0),
        "assets": [],
        "layers": [],
    }

    # Process fonts
    if "fonts" in reconstructed and reconstructed["fonts"]:
        fonts_data = reconstructed["fonts"]
        if isinstance(fonts_data, dict) and "list" in fonts_data:
            fonts_json = {"list": []}
            for font in fonts_data["list"]:
                if isinstance(font, Font):
                    fonts_json["list"].append(font_to_json(font))
                else:
                    fonts_json["list"].append(font)
            json_animation["fonts"] = fonts_json

    # Process chars
    if "chars" in reconstructed and reconstructed["chars"]:
        chars_json = []
        for char in reconstructed["chars"]:
            if isinstance(char, Chars):
                chars_json.append(char_to_json(char))
            else:
                chars_json.append(char)
        json_animation["chars"] = chars_json

    # Process assets
    for asset in reconstructed.get("assets", []):
        asset_json = dict(asset)
        if "layers" in asset:
            asset_json["layers"] = []
            for layer in asset["layers"]:
                if isinstance(layer, ShapeLayer):
                    asset_json["layers"].append(shape_layer_to_json(layer))
                elif isinstance(layer, NullLayer):
                    asset_json["layers"].append(null_layer_to_json(layer))
                elif isinstance(layer, PreCompLayer):
                    asset_json["layers"].append(precomp_layer_to_json(layer))
                elif isinstance(layer, TextLayer):
                    asset_json["layers"].append(text_layer_to_json(layer))
                elif isinstance(layer, SolidColorLayer):
                    asset_json["layers"].append(solid_layer_to_json(layer))
                else:
                    asset_json["layers"].append(layer)
        json_animation["assets"].append(asset_json)

    # Process layers
    for layer in reconstructed.get("layers", []):
        if isinstance(layer, ShapeLayer):
            json_animation["layers"].append(shape_layer_to_json(layer))
        elif isinstance(layer, NullLayer):
            json_animation["layers"].append(null_layer_to_json(layer))
        elif isinstance(layer, PreCompLayer):
            json_animation["layers"].append(precomp_layer_to_json(layer))
        elif isinstance(layer, TextLayer):
            json_animation["layers"].append(text_layer_to_json(layer))
        elif isinstance(layer, SolidColorLayer):
            json_animation["layers"].append(solid_layer_to_json(layer))
        else:
            json_animation["layers"].append(layer)

    return json_animation


def main():
    parser = argparse.ArgumentParser(description='OmniLottie Inference (HF Compatible)')

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                      help='Model path (local path or HF Hub ID, e.g. OmniLottie/OmniLottie)')
    parser.add_argument('--processor_path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct',
                      help='Processor path (local path or HF Hub ID)')

    # Input arguments (choose one)
    parser.add_argument('--text', type=str, help='Text prompt')
    parser.add_argument('--image', type=str, help='Image path')
    parser.add_argument('--video', type=str, help='Video path')

    # Output arguments
    parser.add_argument('--output', type=str, default='output.json',
                      help='Output Lottie JSON file path')

    # Generation arguments
    parser.add_argument('--max_tokens', type=int, default=4096,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--do_sample', action='store_true',
                      help='Enable sampling (otherwise use greedy decoding)')
    parser.add_argument('--temperature', type=float, default=0.9,
                      help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.25,
                      help='Top-p sampling')
    parser.add_argument('--top_k', type=int, default=5,
                      help='Top-k sampling')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Validate input
    if not (args.text or args.image or args.video):
        parser.error("Must provide --text, --image, or --video")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print("="*60)
    print("Loading OmniLottie model...")
    print("="*60)

    print(f"\n1. Loading model from: {args.model_path}")
    model = LottieDecoder.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to(device).eval()
    print(f"   ✓ Model loaded (vocab_size: {model.vocab_size})")

    print(f"\n2. Loading processor from: {args.processor_path}")
    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        padding_side="left",
        trust_remote_code=True
    )
    print(f"   ✓ Processor loaded")

    # Prepare inputs
    print("\n" + "="*60)
    print("Preparing inputs...")
    print("="*60)

    if args.text:
        print(f"\nMode: Text-to-Lottie")
        print(f"Prompt: {args.text}")
        messages = build_messages("text", text_prompt=args.text)

    elif args.image:
        print(f"\nMode: Image-to-Lottie")
        print(f"Image: {args.image}")
        image = Image.open(args.image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((448, 448), Image.LANCZOS)
        messages = build_messages("image", text_prompt=args.text, image=image)

    elif args.video:
        print(f"\nMode: Video-to-Lottie")
        print(f"Video: {args.video}")
        frames = load_frames_from_video(args.video, num_frames=8)
        messages = build_messages("video", video_frames=frames)

    # Prepare inference input
    inputs = prepare_inference_input(processor, messages, device)

    # Generate
    print("\n" + "="*60)
    print("Generating Lottie animation...")
    print("="*60)

    lottie_tokens = generate_lottie(
        model=model,
        inputs=inputs,
        max_tokens=args.max_tokens,
        device=device,
        use_sampling=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )

    print(f"\n✓ Generated {len(lottie_tokens)} Lottie tokens")

    # Convert to JSON
    print("\nConverting tokens to Lottie JSON...")
    lottie_json = tokens_to_lottie_json(lottie_tokens)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(lottie_json, f, indent=2)

    print("\n" + "="*60)
    print("✓ Generation complete!")
    print("="*60)
    print(f"\nOutput saved to: {output_path}")
    print(f"Animation info:")
    print(f"  - Size: {lottie_json['w']} x {lottie_json['h']}")
    print(f"  - Frame rate: {lottie_json['fr']} fps")
    print(f"  - Duration: {lottie_json['op'] - lottie_json['ip']} frames")
    print(f"  - Layers: {len(lottie_json.get('layers', []))}")

    print(f"\n💡 You can now use this Lottie file with:")
    print(f"   - lottie-web: https://airbnb.io/lottie/")
    print(f"   - LottieFiles: https://lottiefiles.com/")


if __name__ == "__main__":
    main()
