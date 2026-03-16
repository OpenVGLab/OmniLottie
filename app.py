import gradio as gr
import json
import torch
import os
import numpy as np
import random
import re
import tempfile
import base64
import threading
import time
from PIL import Image as PILImage
from decord import VideoReader, cpu

from decoder import LottieDecoder
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from lottie.objects.lottie_tokenize import LottieTensor
from lottie.objects.lottie_param import (
    from_sequence, ShapeLayer, NullLayer, PreCompLayer, TextLayer,
    SolidColorLayer, Font, Chars,
    shape_layer_to_json, null_layer_to_json, precomp_layer_to_json,
    text_layer_to_json, solid_layer_to_json, font_to_json, char_to_json
)

SYSTEM_PROMPT = "You are a Lottie animation expert."
VIDEO_PROMPT = "Turn this video into Lottie code."
LOTTIE_BOS = 192398
LOTTIE_EOS = 192399
PAD_TOKEN = 151643

model = None
processor = None
device = None

generation_lock = threading.Lock()

def load_model_once():
    global model, processor, device

    if model is not None:
        return model, processor, device

    checkpoint_path = "/PATH/TO/OmniLottie"


    device = torch.device("cuda:0" if torch.cuda.is_available() else "xpu:0" if torch.xpu.is_available() else "cpu")

    print(f"Loading model from {checkpoint_path}...")
    model = LottieDecoder(pix_len=4560, text_len=1500)

    model_file = os.path.join(checkpoint_path, 'pytorch_model.bin')
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
    else:
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = model.to(device).eval()

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        padding_side="left"
    )

    print(f"✅ Model loaded on {device}")

    return model, processor, device

def simplify_to_animation_description(text):
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

def add_random_background(img):
    if img.mode != 'RGBA':
        return img.convert('RGB')

    light_colors = [(255, 255, 255), (245, 245, 245), (250, 250, 250)]
    bg_color = random.choice(light_colors)
    background = PILImage.new('RGB', img.size, bg_color)
    background.paste(img, (0, 0), img)
    return background

def load_frames_from_video(video_path, num_frames=8, max_size=336):
    import os
    ext = os.path.splitext(video_path)[1].lower()

    frames = []

    if ext in ('.gif', '.webp'):
        try:
            img = PILImage.open(video_path)
            total_frames = getattr(img, 'n_frames', 1)

            if total_frames < 1:
                raise ValueError(f"No frames in {ext.upper()}: {video_path}")

            indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames)).astype(int)

            for idx in indices:
                img.seek(idx)
                frame = img.convert('RGB')
                if max(frame.size) > max_size:
                    frame.thumbnail((max_size, max_size), PILImage.LANCZOS)
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
                img = PILImage.fromarray(f)
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), PILImage.LANCZOS)
                frames.append(img)

        except Exception as e:
            raise ValueError(f"Failed to load video: {str(e)}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames

def build_messages(task_type, text_prompt=None, image=None, video_frames=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if task_type == "text":
        text = simplify_to_animation_description(text_prompt)
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"Generate Lottie code: {text}"}]
        })

    elif task_type == "image":
        text = simplify_to_animation_description(text_prompt)
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

def generate_lottie(model, inputs, max_tokens, device, use_sampling=False, temperature=0.95, top_p=0.25, top_k=5):
    """生成 Lottie tokens"""
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

def fix_lottie_json(anim):

    anim_ip = int(round(anim.get("ip", 0)))
    anim_op = int(round(anim.get("op", 16)))
    anim["ip"]  = anim_ip
    anim["op"]  = anim_op
    anim["fr"]  = int(round(anim.get("fr", 8)))
    anim["ddd"] = int(anim.get("ddd", 0))

    def fix_t_recursive(obj):
        if isinstance(obj, dict):
            if obj.get("a") == 1 and isinstance(obj.get("k"), list):
                for kf in obj["k"]:
                    if isinstance(kf, dict) and "t" in kf:
                        kf["t"] = int(round(kf["t"]))
            for v in obj.values():
                fix_t_recursive(v)
        elif isinstance(obj, list):
            for item in obj:
                fix_t_recursive(item)

    fix_t_recursive(anim)

    max_x = float(anim.get("w", 512))
    max_y = float(anim.get("h", 512))

    def collect_pos(layer):
        nonlocal max_x, max_y
        p = layer.get("ks", {}).get("p", {})
        if isinstance(p, dict):
            if p.get("a", 0) == 0:
                pv = p.get("k", [0, 0])
                if isinstance(pv, list) and len(pv) >= 2:
                    max_x = max(max_x, float(pv[0]))
                    max_y = max(max_y, float(pv[1]))
            else:
                for kf in p.get("k", []):
                    if isinstance(kf, dict):
                        for sv in (kf.get("s", []), kf.get("e", [])):
                            if isinstance(sv, list) and len(sv) >= 2:
                                max_x = max(max_x, float(sv[0]))
                                max_y = max(max_y, float(sv[1]))
        for sub in layer.get("layers", []):
            collect_pos(sub)

    for layer in anim.get("layers", []):
        collect_pos(layer)

    anim["w"] = max(512, int((max_x * 1.1 + 15) // 16 * 16))
    anim["h"] = max(512, int((max_y * 1.1 + 15) // 16 * 16))

    valid_inds = set()
    for layer in anim.get("layers", []):
        if "ind" in layer:
            valid_inds.add(int(layer["ind"]))

    def clean_shapes(shapes):
        if not isinstance(shapes, list):
            return shapes
        cleaned = []
        for sh in shapes:
            if not isinstance(sh, dict):
                continue
            if sh.get("ty") == "gr":
                sh["it"] = clean_shapes(sh.get("it", []))
                if not sh["it"]:
                    continue
                has_tr = any(item.get("ty") == "tr" for item in sh["it"] if isinstance(item, dict))
                if not has_tr:
                    sh["it"].append({
                        "ty": "tr", "nm": "",
                        "a": {"a": 0, "k": [0, 0], "ix": 1},
                        "p": {"a": 0, "k": [0, 0], "ix": 2},
                        "s": {"a": 0, "k": [100, 100], "ix": 3},
                        "r": {"a": 0, "k": 0, "ix": 6},
                        "o": {"a": 0, "k": 100, "ix": 7},
                        "sk": {"a": 0, "k": 0, "ix": 4},
                        "sa": {"a": 0, "k": 0, "ix": 5},
                        "hd": False
                    })
            cleaned.append(sh)
        return cleaned

    def fix_layer(layer):
        ip = int(round(layer.get("ip", anim_ip)))
        op = int(round(layer.get("op", anim_op)))
        layer["ip"] = max(anim_ip, ip)
        layer["op"] = min(anim_op, max(layer["ip"] + 1, op))
        layer["st"] = int(round(layer.get("st", anim_ip)))
        if "ind" in layer:
            layer["ind"] = int(layer["ind"])
        if "parent" in layer:
            p = int(layer["parent"])
            if p in valid_inds:
                layer["parent"] = p
            else:
                del layer["parent"]
        layer.pop("ct", None)
        if "shapes" in layer:
            layer["shapes"] = clean_shapes(layer["shapes"])
        for sub in layer.get("layers", []):
            fix_layer(sub)
        return layer

    fixed_layers = []
    for l in anim.get("layers", []):
        fix_layer(l)
        shapes = l.get("shapes", [])
        if l.get("ty") == 4 and not shapes:
            continue  
        fixed_layers.append(l)
    anim["layers"] = fixed_layers

    for asset in anim.get("assets", []):
        if "layers" in asset:
            fixed = []
            for l in asset["layers"]:
                fix_layer(l)
                if l.get("ty") == 4 and not l.get("shapes"):
                    continue
                fixed.append(l)
            asset["layers"] = fixed

    return anim

def tokens_to_lottie_json(generated_ids):
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

    if "chars" in reconstructed and reconstructed["chars"]:
        chars_json = []
        for char in reconstructed["chars"]:
            if isinstance(char, Chars):
                chars_json.append(char_to_json(char))
            else:
                chars_json.append(char)
        json_animation["chars"] = chars_json

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

    json_animation = fix_lottie_json(json_animation)

    return json_animation

_lottie_js_cache = None

def get_lottie_js():
    global _lottie_js_cache
    if _lottie_js_cache is not None:
        return _lottie_js_cache

    local_path = "lottie.min.js"
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8') as f:
            _lottie_js_cache = f.read()
        print(f"✅ Loaded local lottie.min.js ({len(_lottie_js_cache)} bytes)")
    else:
        _lottie_js_cache = '<script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script>'
        print("⚠️ Using CDN lottie.min.js (local file not found)")

    return _lottie_js_cache

def create_lottie_html(animation_data, height=600):
    bg_style = """
        background-image:
            linear-gradient(45deg, #666666 25%, transparent 25%),
            linear-gradient(-45deg, #666666 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #666666 75%),
            linear-gradient(-45deg, transparent 75%, #666666 75%);
        background-size: 16px 16px;
        background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
        background-color: #444444;
    """

    lottie_js = get_lottie_js()

    if lottie_js.startswith('<script'):
        lottie_script = lottie_js
    else:
        lottie_script = f"<script>{lottie_js}</script>"

    animation_json_escaped = json.dumps(animation_data).replace('\\', '\\\\').replace("'", "\\'")

    anim_width = animation_data.get('w', 512)
    anim_height = animation_data.get('h', 512)

    inner_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    {lottie_script}
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: transparent;
        }}
        #lottie-container {{
            width: 100%;
            height: 100%;
            {bg_style}
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        #lottie-animation {{
            max-width: 100%;
            max-height: 100%;
            width: {anim_width}px;
            height: {anim_height}px;
        }}
        .error {{ color: white; text-align: center; padding: 20px; }}
    </style>
</head>
<body>
    <div id="lottie-container">
        <div id="lottie-animation"></div>
    </div>
    <script>
        function renderLottie() {{
            try {{
                if (typeof lottie === 'undefined') {{
                    setTimeout(renderLottie, 100);
                    return;
                }}
                var animationData = JSON.parse('{animation_json_escaped}');
                lottie.loadAnimation({{
                    container: document.getElementById('lottie-animation'),
                    renderer: 'svg',
                    loop: true,
                    autoplay: true,
                    animationData: animationData
                }});
            }} catch (e) {{
                console.error('Lottie render error:', e);
                document.getElementById('lottie-container').innerHTML =
                    '<p class="error">Failed: ' + e.message + '</p>';
            }}
        }}
        renderLottie();
    </script>
</body>
</html>"""

    inner_html_b64 = base64.b64encode(inner_html.encode('utf-8')).decode('utf-8')
    iframe_html = f'<iframe src="data:text/html;base64,{inner_html_b64}" style="width:100%; height:{height}px; border:none; border-radius:8px;"></iframe>'

    return iframe_html

def save_json_to_temp(lottie_json):
    if lottie_json is None:
        return None

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='lottie_')
    json.dump(lottie_json, temp_file, indent=2)
    temp_file.close()
    return temp_file.name

def process_text_to_lottie(text_prompt, max_tokens, use_sampling, temperature, top_p, top_k):
    with generation_lock:
        try:
            start_time = time.time()

            if not text_prompt or not text_prompt.strip():
                return None, "❌ Please enter a text description", None
            model, processor, device = load_model_once()

            messages = build_messages("text", text_prompt=text_prompt)
            inputs = prepare_inference_input(processor, messages, device)

            generated_ids = generate_lottie(
                model, inputs, max_tokens, device,
                use_sampling, temperature, top_p, top_k
            )

            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.xpu.is_available():
                torch.xpu.empty_cache()

            lottie_json = tokens_to_lottie_json(generated_ids)

            html = create_lottie_html(lottie_json, height=600)

            elapsed_time = time.time() - start_time

            status = f"✅ Generated {len(generated_ids)} tokens | Layers: {len(lottie_json.get('layers', []))} | {lottie_json.get('fr', 8)} fps | Time: {elapsed_time:.1f}s"

            temp_path = save_json_to_temp(lottie_json)

            return html, status, temp_path

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.xpu.is_available():
                torch.xpu.empty_cache()
            return None, f"❌ Error: {str(e)}", None

def load_image_from_file(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.svg':
        import cairosvg
        import io
        png_bytes = cairosvg.svg2png(url=file_path, output_width=448, output_height=448)
        image = PILImage.open(io.BytesIO(png_bytes))
    else:
        image = PILImage.open(file_path)

    if image.mode == 'RGBA':
        image = add_random_background(image)
    else:
        image = image.convert('RGB')

    return image


def process_image_to_lottie(image_file, text_description, max_tokens, use_sampling, temperature, top_p, top_k):
    with generation_lock:
        try:
            start_time = time.time()

            if image_file is None:
                return None, "❌ Please upload an image", None

            model, processor, device = load_model_once()

            image = load_image_from_file(image_file)

            image = image.resize((448, 448), PILImage.LANCZOS)

            desc = text_description if text_description else "A simple animation"
            messages = build_messages("image", text_prompt=desc, image=image)
            inputs = prepare_inference_input(processor, messages, device)

            generated_ids = generate_lottie(
                model, inputs, max_tokens, device,
                use_sampling, temperature, top_p, top_k
            )

            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.xpu.is_available():
                torch.xpu.empty_cache()

            lottie_json = tokens_to_lottie_json(generated_ids)

            html = create_lottie_html(lottie_json, height=600)

            elapsed_time = time.time() - start_time

            status = f"✅ Generated {len(generated_ids)} tokens | Layers: {len(lottie_json.get('layers', []))} | {lottie_json.get('fr', 8)} fps | Time: {elapsed_time:.1f}s"

            temp_path = save_json_to_temp(lottie_json)

            return html, status, temp_path

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.xpu.is_available():
                torch.xpu.empty_cache()
            return None, f"❌ Error: {str(e)}", None

def process_video_to_lottie(video, max_tokens, use_sampling, temperature, top_p, top_k):
    with generation_lock:
        try:
            start_time = time.time()

            if video is None:
                return None, "❌ Please upload a video/GIF/WebP file", None

            import os
            ext = os.path.splitext(video)[1].lower() if isinstance(video, str) else ''
            if ext not in ('.mp4', '.avi', '.mov', '.gif', '.webp'):
                return None, f"❌ Unsupported format: {ext}. Please upload MP4/AVI/MOV/GIF/WebP", None

            model, processor, device = load_model_once()

            frames = load_frames_from_video(video, num_frames=8)

            messages = build_messages("video", video_frames=frames)
            inputs = prepare_inference_input(processor, messages, device)

            generated_ids = generate_lottie(
                model, inputs, max_tokens, device,
                use_sampling, temperature, top_p, top_k
            )

            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.xpu.is_available():
                torch.xpu.empty_cache()

            lottie_json = tokens_to_lottie_json(generated_ids)

            html = create_lottie_html(lottie_json, height=600)

            elapsed_time = time.time() - start_time

            status = f"✅ Generated {len(generated_ids)} tokens (from {len(frames)} frames) | Layers: {len(lottie_json.get('layers', []))} | {lottie_json.get('fr', 8)} fps | Time: {elapsed_time:.1f}s"

            temp_path = save_json_to_temp(lottie_json)

            return html, status, temp_path

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.xpu.is_available():
                torch.xpu.empty_cache()
            return None, f"❌ Error: {str(e)}", None

def create_gradio_interface():

    with gr.Blocks(title="OmniLottie Demo Page", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 OmniLottie Demo Page")
        gr.Markdown("Offical Demo Page of OmniLottie")
        gr.Markdown("Generate Lottie animations from text, images, or videos")

        with gr.Tabs() as tabs:
            with gr.Tab("📝 Text-to-Lottie"):
                gr.Markdown("""
                ### 💡 Prompt Tips for Better Results

                **Good prompts should describe:**
                1. **Main Object**: What is being animated (e.g., "a blue bird", "a yellow folder icon", "an orange piggy bank")
                2. **Motion Pattern**: How it moves (e.g., "appearing, pulsing while sliding", "fading in, floating toward", "bouncing up and down")
                3. **Direction**: Where it moves (e.g., "downward", "toward the top-left", "back to its start")
                4. **Loop Behavior**: How it repeats (e.g., "repeating seamlessly", "looping smoothly", "repeating continuously")

                **Example Patterns:**
                - 🔄 **Simple Loop**: "a red ball appearing, bouncing up and down, then fading out, repeating seamlessly"
                - 🎯 **Movement**: "a blue arrow sliding from left to right, then quickly returning to start, looping continuously"
                - 💫 **Transformation**: "a yellow star fading in while rotating 360 degrees, holds briefly, then fading out, repeating smoothly"
                - 🎨 **Static Icon**: "static illustration of a cartoon cat's face with a cute expression, light orange body, red inner ears"
                - 👤 **Character**: "animated cartoon figure dressed in a beige suit with a white shirt, holding a gray tablet"

                **Pro Tips:**
                - Be specific about colors, shapes, and movements
                - Describe motion phases clearly (appear → move → hold → return)
                - Use descriptive motion verbs: sliding, pulsing, drifting, bouncing, rotating, fading
                - For icons: include style details (outline, colors, decorations)
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="Text Description",
                            placeholder="Example: a blue bird appearing, pulsing while sliding downward, lingers briefly, then growing back while sliding upward to reset, repeating seamlessly",
                            lines=5
                        )

                        with gr.Accordion("⚙️ Generation Settings", open=False):
                            gr.Markdown("""
                            **Parameter Guide:**
                            - **Max Tokens**: Higher = more complex animations (slower), Lower = simpler animations (faster)
                            - **Top-p & Top-k**: Higher = more random/creative, Lower = more stable/consistent
                            - **Temperature**: Higher = more diverse, Lower = more deterministic

                            💡 **Quick Tips:**
                            - For complex animations: increase max tokens to 5856
                            - For faster generation: reduce max tokens to 2048-3072
                            - For more creative results: increase top-p (0.5-0.8) and top-k (20-50)
                            - For consistent results: decrease top-p (0.1-0.25) and top-k (5-10)
                            """)
                            text_max_tokens = gr.Slider(512, 5856, value=5556, step=256, label="Max Tokens")
                            text_use_sampling = gr.Checkbox(label="Use Sampling", value=True)
                            text_temperature = gr.Slider(0.1, 2.0, value=0.9, step=0.1, label="Temperature")
                            text_top_p = gr.Slider(0.1, 1.0, value=0.25, step=0.1, label="Top-p")
                            text_top_k = gr.Slider(1, 100, value=5, step=1, label="Top-k")

                        text_generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

                        # Generation time tips
                        gr.Markdown("""
                        ⏱️ **Generation Time:**
                        - Simple icons/shapes: ~30-60 seconds (1000-2000 tokens)
                        - Medium animations: ~1-2 minutes (2000-3500 tokens)
                        - Complex characters: ~4-5 minutes (4500-6000 tokens)

                        Please be patient! Complex animations take time to generate. ☕
                        """)

                        text_status = gr.Markdown()

                    with gr.Column(scale=1):
                        text_output = gr.HTML(label="Animation Preview")
                        text_json_file = gr.File(label="📥 Download JSON", visible=True)

                def get_text_examples():
                    examples = []
                    demo_txt_path = "./example/demo.txt"
                    if os.path.exists(demo_txt_path):
                        with open(demo_txt_path, 'r', encoding='utf-8') as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                            examples = [[line] for line in lines[:50]]  
                    return examples

                gr.Examples(
                    examples=get_text_examples(),
                    inputs=text_input,
                    label="📂 Example Prompts (Click to Load)",
                    examples_per_page=10,
                    cache_examples=False
                )

                text_generate_btn.click(
                    fn=process_text_to_lottie,
                    inputs=[text_input, text_max_tokens, text_use_sampling, text_temperature, text_top_p, text_top_k],
                    outputs=[text_output, text_status, text_json_file]
                )

            with gr.Tab("🖼️ Text+Image-to-Lottie"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Image",
                            type="filepath",
                            sources=["upload"]
                        )
                        image_text_input = gr.Textbox(
                            label="Animation Description",
                            placeholder="Example: The object rotates 360 degrees",
                            lines=3
                        )

                        with gr.Accordion("⚙️ Generation Settings", open=False):
                            gr.Markdown("""
                            **Parameter Guide:**
                            - **Max Tokens**: Higher = more complex animations (slower), Lower = simpler (faster)
                            - **Top-p & Top-k**: Higher = more creative/random, Lower = more stable
                            - **Temperature**: Controls output diversity
                            """)
                            image_max_tokens = gr.Slider(512, 5556, value=5556, step=256, label="Max Tokens")
                            image_use_sampling = gr.Checkbox(label="Use Sampling", value=True)
                            image_temperature = gr.Slider(0.1, 2.0, value=0.9, step=0.1, label="Temperature")
                            image_top_p = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Top-p")
                            image_top_k = gr.Slider(1, 100, value=5, step=1, label="Top-k")

                        image_generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

                        # Generation time tips
                        gr.Markdown("""
                        ⏱️ **Generation Time:** ~1-5 minutes depending on complexity
                        """)

                        image_status = gr.Markdown()

                    with gr.Column(scale=1):
                        image_output = gr.HTML(label="Animation Preview")
                        image_json_file = gr.File(label="📥 Download JSON", visible=True)

                def get_image_text_examples():
                    examples = []
                    demo_images_dir = "./example/demo_images"
                    if os.path.exists(demo_images_dir):
                        png_files = sorted([f for f in os.listdir(demo_images_dir) if f.endswith('.png')])
                        for png_file in png_files[:50]:  
                            base_name = os.path.splitext(png_file)[0]
                            txt_file = os.path.join(demo_images_dir, f"{base_name}.txt")
                            png_path = os.path.join(demo_images_dir, png_file)
                            if os.path.exists(txt_file):
                                with open(txt_file, 'r', encoding='utf-8') as f:
                                    text_desc = f.read().strip()
                                examples.append([png_path, text_desc])
                    return examples

                gr.Examples(
                    examples=get_image_text_examples(),
                    inputs=[image_input, image_text_input],
                    label="📂 Example Images (Click to Load)",
                    examples_per_page=5,
                    cache_examples=False
                )

                image_generate_btn.click(
                    fn=process_image_to_lottie,
                    inputs=[image_input, image_text_input, image_max_tokens, image_use_sampling,
                           image_temperature, image_top_p, image_top_k],
                    outputs=[image_output, image_status, image_json_file]
                )

            # Tab 3: Video-to-Lottie
            with gr.Tab("🎥 Video-to-Lottie"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Upload Video / GIF / WebP",
                            sources=["upload"]
                        )

                        with gr.Accordion("⚙️ Generation Settings", open=False):
                            gr.Markdown("""
                            **Parameter Guide:**
                            - **Max Tokens**: Higher = more complex animations (slower), Lower = simpler (faster)
                            - **Top-p & Top-k**: Higher = more creative/random, Lower = more stable
                            - **Temperature**: Controls output diversity
                            """)
                            video_max_tokens = gr.Slider(512, 5556, value=5556, step=256, label="Max Tokens")
                            video_use_sampling = gr.Checkbox(label="Use Sampling", value=True)
                            video_temperature = gr.Slider(0.1, 2.0, value=0.9, step=0.1, label="Temperature")
                            video_top_p = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Top-p")
                            video_top_k = gr.Slider(1, 100, value=5, step=1, label="Top-k")

                        video_generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

                        # Generation time tips
                        gr.Markdown("""
                        ⏱️ **Generation Time:** ~2-5 minutes depending on video complexity
                        """)

                        video_status = gr.Markdown()

                    with gr.Column(scale=1):
                        video_output = gr.HTML(label="Animation Preview")
                        video_json_file = gr.File(label="📥 Download JSON", visible=True)

                def get_video_examples():
                    examples = []
                    demo_video_dir = "./example/demo_video"
                    if os.path.exists(demo_video_dir):
                        video_files = sorted([os.path.join(demo_video_dir, f)
                                            for f in os.listdir(demo_video_dir)
                                            if f.endswith('.mp4')])
                        examples = [[vf] for vf in video_files[:50]]  
                    return examples

                gr.Examples(
                    examples=get_video_examples(),
                    inputs=video_input,
                    label="📂 Example Videos (Click to Load)",
                    examples_per_page=5,
                    cache_examples=False
                )

                video_generate_btn.click(
                    fn=process_video_to_lottie,
                    inputs=[video_input, video_max_tokens, video_use_sampling,
                           video_temperature, video_top_p, video_top_k],
                    outputs=[video_output, video_status, video_json_file]
                )

        gr.Markdown("---")
        gr.Markdown("""
        ⚠️ **Important Note:** This demo processes one request at a time.
        If another user is generating, your request will wait in queue until the current one completes.
        """)

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
