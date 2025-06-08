import os
import sys
sys.path.append(os.getcwd())
import glob
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import gradio as gr
import time
import uuid
import gc
import requests
import argparse
import base64
from io import BytesIO

# 导入原有代码中的所有函数
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

# 保留原始代码中的辅助函数
tensor_transforms = transforms.Compose([transforms.ToTensor()])
ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def resize_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))

def call_litellm_api(messages, api_base, api_key, model="gpt-4"):  # 默认gpt-4，可自定义
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0.7
    }
    response = requests.post(f"{api_base}/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# 从原始代码中保留get_validation_prompt函数
def get_validation_prompt(args, image, prompt_image_path, dape_model=None, vlm_model_in=None, device='cuda'):
    global vlm_model, vlm_processor, process_vision_info
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    if getattr(args, 'prompt_backend', 'local') == 'litellm':
        # 支持图片base64
        if args.rec_type == "recursive_multiscale":
            user_prompt = "The second image is a zoom-in of the first image. Based on this knowledge, what is in the second image? Give me a set of words."
        else:
            user_prompt = "What is in this image? Give me a set of words."
        img_b64 = image_to_base64(image)
        messages = [
            {"role": "system", "content": user_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "请根据图像内容生成一组英文关键词。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}
        ]
        prompt_text = call_litellm_api(
            messages,
            api_base=getattr(args, 'litellm_api_base', ''),
            api_key=getattr(args, 'litellm_api_key', ''),
            model=getattr(args, 'litellm_model', 'gpt-4')
        )
        if args.prompt:
            prompt_text = f"{prompt_text}, {args.prompt}"
        return prompt_text, lq
    # 本地VLM分支，按需加载
    if args.prompt_type in ("vlm") and getattr(args, 'prompt_backend', 'local') == 'local':
        if vlm_model is None or vlm_processor is None or process_vision_info is None:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info as pvi
            vlm_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            print(f"[LazyLoad] Loading base VLM model: {vlm_model_name}")
            vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vlm_model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
            process_vision_info = pvi
        message_text = None
        
        if args.rec_type == "recursive":
            message_text = "What is in this image? Give me a set of words."
            print(f'MESSAGE TEXT: {message_text}')
            messages = [
                {"role": "system", "content": f"{message_text}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": prompt_image_path}
                    ]
                }
            ]
            text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
        elif args.rec_type == "recursive_multiscale":
            start_image_path = prompt_image_path[0]
            input_image_path = prompt_image_path[1]
            message_text = "The second image is a zoom-in of the first image. Based on this knowledge, what is in the second image? Give me a set of words."
            print(f'START IMAGE PATH: {start_image_path}\nINPUT IMAGE PATH: {input_image_path}\nMESSAGE TEXT: {message_text}')
            messages = [
                {"role": "system", "content": f"{message_text}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": start_image_path},
                        {"type": "image", "image": input_image_path}
                    ]
                }
            ]
            print(f'MESSAGES\n{messages}')

            text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        else:
            raise ValueError(f"VLM prompt generation not implemented for rec_type: {args.rec_type}")

        inputs = inputs.to("cuda")

        original_sr_devices = {}
        if args.efficient_memory and 'model' in globals() and hasattr(model, 'text_enc_1'): # Check if SR model is defined
            print("Moving SR model components to CPU for VLM inference.")
            original_sr_devices['text_enc_1'] = model.text_enc_1.device
            original_sr_devices['text_enc_2'] = model.text_enc_2.device
            original_sr_devices['text_enc_3'] = model.text_enc_3.device
            original_sr_devices['transformer'] = model.transformer.device
            original_sr_devices['vae'] = model.vae.device
            
            model.text_enc_1.to('cpu')
            model.text_enc_2.to('cpu')
            model.text_enc_3.to('cpu')
            model.transformer.to('cpu')
            model.vae.to('cpu')
            vlm_model.to('cuda') # vlm_model should already be on its device_map="auto" device

        generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        prompt_text = f"{output_text[0]}, {args.prompt}," if args.prompt else output_text[0]

        if args.efficient_memory and 'model' in globals() and hasattr(model, 'text_enc_1'):
            print("Restoring SR model components to original devices.")
            vlm_model.to('cpu') # If vlm_model was moved to a specific cuda device and needs to be offloaded
            model.text_enc_1.to(original_sr_devices['text_enc_1'])
            model.text_enc_2.to(original_sr_devices['text_enc_2'])
            model.text_enc_3.to(original_sr_devices['text_enc_3'])
            model.transformer.to(original_sr_devices['transformer'])
            model.vae.to(original_sr_devices['vae'])
    elif args.prompt_type == "null":
        prompt_text = args.prompt or ""
    elif args.prompt_type == "dape":
        lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
        captions = inference(lq_ram, dape_model)
        prompt_text = f"{captions[0]}, {args.prompt}," if args.prompt else captions[0]
    else:
        raise ValueError(f"Unknown prompt_type: {args.prompt_type}")
    return prompt_text, lq

# 创建增强型拼接图像函数
def create_enhanced_concat(images, layout="horizontal", spacing=5, background_color=(240, 240, 240)):
    """
    创建增强型拼接图像，支持水平、垂直或网格布局
    
    Args:
        images: 图像列表
        layout: 布局方式，可选 "horizontal"、"vertical" 或 "grid"
        spacing: 图像间距
        background_color: 背景颜色
    
    Returns:
        拼接后的图像
    """
    if not images:
        return None
        
    if layout == "horizontal":
        # 水平拼接
        total_width = sum(img.width for img in images) + spacing * (len(images) - 1)
        max_height = max(img.height for img in images)
        
        concat = Image.new('RGB', (total_width, max_height), background_color)
        x_offset = 0
        
        for img in images:
            concat.paste(img, (x_offset, (max_height - img.height) // 2))
            x_offset += img.width + spacing
            
    elif layout == "vertical":
        # 垂直拼接
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images) + spacing * (len(images) - 1)
        
        concat = Image.new('RGB', (max_width, total_height), background_color)
        y_offset = 0
        
        for img in images:
            concat.paste(img, ((max_width - img.width) // 2, y_offset))
            y_offset += img.height + spacing
            
    elif layout == "grid":
        # 网格布局（尝试创建接近正方形的布局）
        n = len(images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # 计算每个图像的目标大小（保持宽高比）
        base_size = min(images[0].width, images[0].height)
        scaled_images = []
        
        for img in images:
            # 保持宽高比的缩放
            ratio = base_size / min(img.width, img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            scaled_images.append(img.resize(new_size, Image.LANCZOS))
        
        cell_width = max(img.width for img in scaled_images) + spacing
        cell_height = max(img.height for img in scaled_images) + spacing
        
        grid_width = cols * cell_width - spacing
        grid_height = rows * cell_height - spacing
        
        concat = Image.new('RGB', (grid_width, grid_height), background_color)
        
        for idx, img in enumerate(scaled_images):
            row = idx // cols
            col = idx % cols
            
            x = col * cell_width + (cell_width - img.width) // 2
            y = row * cell_height + (cell_height - img.height) // 2
            
            concat.paste(img, (x, y))
    else:
        raise ValueError(f"不支持的布局类型: {layout}")
        
    return concat

# ========== 全局模型加载 ==========
# 只加载一次，后续推理直接复用

# 配置参数（可根据需要调整）
MODEL_CONFIG = {
    'process_size': 512,
    'upscale': 4,
    'rec_type': 'recursive_multiscale',
    'rec_num': 4,
    'prompt_type': 'vlm',
    'align_method': 'nofix',
    'user_prompt': '',
    'efficient_memory': True,
    'mixed_precision': 'fp16',
    'concat_layout': 'horizontal',
    'lora_path': 'ckpt/SR_LoRA/model_20001.pkl',
    'vae_path': 'ckpt/SR_VAE/vae_encoder_20001.pt',
    'pretrained_model_name_or_path': 'stabilityai/stable-diffusion-3-medium-diffusers',
    'ram_ft_path': 'ckpt/DAPE/DAPE.pth',
    'ram_path': 'ckpt/RAM/ram_swin_large_14m.pth',
    'save_prompts': True,
    'merge_and_unload_lora': False,
    'lora_rank': 4,
    'vae_decoder_tiled_size': 224,
    'vae_encoder_tiled_size': 1024,
    'latent_tiled_size': 96,
    'latent_tiled_overlap': 32,
    'seed': 42
}

class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # 新增默认参数
        self.prompt_backend = kwargs.get('prompt_backend', 'local')
        self.litellm_api_base = kwargs.get('litellm_api_base', '')
        self.litellm_api_key = kwargs.get('litellm_api_key', '')
        self.litellm_model = kwargs.get('litellm_model', 'gpt-4')

args_global = Args(**MODEL_CONFIG)
global weight_dtype
weight_dtype = torch.float16 if args_global.mixed_precision == "fp16" else torch.float32

global model, model_test, DAPE
model = None
model_test = None
DAPE = None

# 加载SR模型
if args_global.rec_type not in ('nearest', 'bicubic'):
    if not args_global.efficient_memory:
        from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler
        model = SD3Euler()
        model.text_enc_1.to('cuda')
        model.text_enc_2.to('cuda')
        model.text_enc_3.to('cuda')
        model.transformer.to('cuda', dtype=torch.float32)
        model.vae.to('cuda', dtype=torch.float32)
        for p in [model.text_enc_1, model.text_enc_2, model.text_enc_3, model.transformer, model.vae]:
            p.requires_grad_(False)
        model_test = OSEDiff_SD3_TEST(args_global, model)
    else:
        from osediff_sd3 import OSEDiff_SD3_TEST_efficient, SD3Euler
        model = SD3Euler()
        model.transformer.to('cuda', dtype=torch.float32)
        model.vae.to('cuda', dtype=torch.float32)
        for p in [model.text_enc_1, model.text_enc_2, model.text_enc_3, model.transformer, model.vae]:
            p.requires_grad_(False)
        model_test = OSEDiff_SD3_TEST_efficient(args_global, model)

# 加载DAPE模型（如需要）
if args_global.prompt_type == "dape":
    DAPE = ram(pretrained=args_global.ram_path,
              pretrained_condition=args_global.ram_ft_path,
              image_size=384,
              vit='swin_l')
    DAPE.eval().to("cuda")
    DAPE = DAPE.to(dtype=weight_dtype)

# 主处理函数，将被UI调用
def process_image(
    input_image,
    process_size=512,
    upscale=4,
    rec_type="recursive_multiscale",
    rec_num=4,
    prompt_type="vlm",
    align_method="nofix",
    user_prompt="",
    efficient_memory=True,
    mixed_precision="fp16",
    concat_layout="horizontal",
    prompt_backend="local",
    litellm_api_base="",
    litellm_api_key="",
    litellm_model="gpt-4"
):
    global model, model_test, DAPE
    # 创建临时目录存储结果
    temp_dir = "temp_results"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'per-sample'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'per-scale'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'recursive'), exist_ok=True)
    
    # 生成唯一ID作为结果标识
    result_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # 保存输入图像
    input_path = os.path.join(temp_dir, f"input_{result_id}.png")
    input_image.save(input_path)
    
    # 创建参数对象
    class Args:
        def __init__(self):
            self.process_size = process_size
            self.upscale = upscale
            self.rec_type = rec_type
            self.rec_num = rec_num
            self.prompt_type = prompt_type
            self.align_method = align_method
            self.prompt = user_prompt
            self.efficient_memory = efficient_memory
            self.mixed_precision = mixed_precision
            self.output_dir = temp_dir
            self.input_image = input_path
            self.lora_path = "ckpt/SR_LoRA/model_20001.pkl"
            self.vae_path = "ckpt/SR_VAE/vae_encoder_20001.pt"
            self.pretrained_model_name_or_path = "stabilityai/stable-diffusion-3-medium-diffusers"
            self.ram_ft_path = "ckpt/DAPE/DAPE.pth"
            self.ram_path = "ckpt/RAM/ram_swin_large_14m.pth"
            self.save_prompts = True
            self.merge_and_unload_lora = False
            self.lora_rank = 4
            self.vae_decoder_tiled_size = 224
            self.vae_encoder_tiled_size = 1024
            self.latent_tiled_size = 96
            self.latent_tiled_overlap = 32
            self.seed = 42
            # 新增
            self.prompt_backend = prompt_backend
            self.litellm_api_base = litellm_api_base
            self.litellm_api_key = litellm_api_key
            self.litellm_model = litellm_model
    
    args = Args()
    
    global weight_dtype
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    
    # 处理图像
    bname = f"result_{result_id}.png"
    rec_dir = os.path.join(args.output_dir, 'per-sample', result_id)
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'per-scale', 'scale0'), exist_ok=True)
    
    # 第一张图像处理
    first_image = resize_and_center_crop(input_image, args.process_size)
    first_image.save(f'{rec_dir}/0.png')
    first_image.save(os.path.join(args.output_dir, 'per-scale', 'scale0', bname))
    
    # 递归处理
    results = []
    results.append(first_image)
    
    # 记录每一步的提示词
    prompts = []
    prompts.append("原始图像")
    
    for rec in range(args.rec_num):
        print(f'RECURSION: {rec}')
        os.makedirs(os.path.join(args.output_dir, 'per-scale', f'scale{rec+1}'), exist_ok=True)
        
        # 根据递归类型处理图像
        if rec_type in ('nearest', 'bicubic', 'onestep'):
            # 处理代码与原代码相同
            start_image_pil_path = f'{rec_dir}/0.png'
            start_image_pil = Image.open(start_image_pil_path).convert('RGB')
            rscale = pow(args.upscale, rec+1)
            w, h = start_image_pil.size
            new_w, new_h = w // rscale, h // rscale
            
            cropped_region = start_image_pil.crop(((w-new_w)//2, (h-new_h)//2, (w+new_w)//2, (h+new_h)//2))
            
            if rec_type == 'onestep':
                current_sr_input_image_pil = cropped_region.resize((w, h), Image.BICUBIC)
                prompt_image_path = f'{rec_dir}/0_input_for_{rec+1}.png'
                current_sr_input_image_pil.save(prompt_image_path)
            elif rec_type == 'bicubic':
                current_sr_input_image_pil = cropped_region.resize((w, h), Image.BICUBIC)
                current_sr_input_image_pil.save(f'{rec_dir}/{rec+1}.png')
                results.append(current_sr_input_image_pil)
                prompts.append(f"双三次插值 #{rec+1}")
                continue
            elif rec_type == 'nearest':
                current_sr_input_image_pil = cropped_region.resize((w, h), Image.NEAREST)
                current_sr_input_image_pil.save(f'{rec_dir}/{rec+1}.png')
                results.append(current_sr_input_image_pil)
                prompts.append(f"最近邻插值 #{rec+1}")
                continue
                
        elif rec_type == 'recursive':
            # 处理代码与原代码相同
            prev_sr_output_path = f'{rec_dir}/{rec}.png'
            prev_sr_output_pil = Image.open(prev_sr_output_path).convert('RGB')
            rscale = args.upscale
            w, h = prev_sr_output_pil.size
            new_w, new_h = w // rscale, h // rscale
            cropped_region = prev_sr_output_pil.crop(((w-new_w)//2, (h-new_h)//2, (w+new_w)//2, (h+new_h)//2))
            current_sr_input_image_pil = cropped_region.resize((w, h), Image.BICUBIC)
            
            input_image_path = f'{rec_dir}/{rec+1}_input.png'
            current_sr_input_image_pil.save(input_image_path)
            prompt_image_path = input_image_path
            
        elif rec_type == 'recursive_multiscale':
            # 处理代码与原代码相同
            prev_sr_output_path = f'{rec_dir}/{rec}.png'
            prev_sr_output_pil = Image.open(prev_sr_output_path).convert('RGB')
            rscale = args.upscale
            w, h = prev_sr_output_pil.size
            new_w, new_h = w // rscale, h // rscale
            cropped_region = prev_sr_output_pil.crop(((w-new_w)//2, (h-new_h)//2, (w+new_w)//2, (h+new_h)//2))
            current_sr_input_image_pil = cropped_region.resize((w, h), Image.BICUBIC)
            
            zoomed_image_path = f'{rec_dir}/{rec+1}_input.png'
            current_sr_input_image_pil.save(zoomed_image_path)
            prompt_image_path = [prev_sr_output_path, zoomed_image_path]
            
        # 生成提示词
        validation_prompt, lq = get_validation_prompt(args, current_sr_input_image_pil, prompt_image_path, DAPE)
        print(f'TAG: {validation_prompt}')
        
        # 保存本次使用的提示词
        prompts.append(f"步骤 #{rec+1}: {validation_prompt[:50]}..." if len(validation_prompt) > 50 else f"步骤 #{rec+1}: {validation_prompt}")
        
        # 超分辨率处理
        with torch.no_grad():
            lq = lq * 2 - 1
            
            if args.efficient_memory and model is not None:
                print("Ensuring SR model components are on CUDA for SR inference.")
                model.transformer.to('cuda', dtype=torch.float32)
                model.vae.to('cuda', dtype=torch.float32)
                
            output_image = model_test(lq, prompt=validation_prompt)
            output_image = torch.clamp(output_image[0].cpu(), -1.0, 1.0)
            output_pil = transforms.ToPILImage()(output_image * 0.5 + 0.5)
            if args.align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=current_sr_input_image_pil)
            elif args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=current_sr_input_image_pil)
                
        output_pil.save(f'{rec_dir}/{rec+1}.png')
        results.append(output_pil)
        
    # 创建拼接图像，使用增强的拼接函数
    concat = create_enhanced_concat(results, layout=concat_layout)
    concat_path = os.path.join(args.output_dir, 'recursive', bname)
    concat.save(concat_path)
    
    # 获取最终图像尺寸信息
    final_image = results[-1]
    size_info = f"{final_image.width} x {final_image.height} 像素"
    
    # 获取每一步的放大区域
    zoom_regions = []
    
    # 计算每一步放大的区域（用于可视化）
    if rec_type in ('recursive', 'recursive_multiscale'):
        for i in range(rec_num):
            if i == 0:
                # 第一步的原始图像
                orig_img = results[0].copy()
                w, h = orig_img.size
                rscale = args.upscale
                new_w, new_h = w // rscale, h // rscale
                left, top = (w - new_w) // 2, (h - new_h) // 2
                right, bottom = left + new_w, top + new_h
                
                # 在原图上标记放大区域
                from PIL import ImageDraw
                draw = ImageDraw.Draw(orig_img)
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                zoom_regions.append(orig_img)
            else:
                # 后续步骤的图像
                orig_img = results[i].copy()
                w, h = orig_img.size
                rscale = args.upscale
                new_w, new_h = w // rscale, h // rscale
                left, top = (w - new_w) // 2, (h - new_h) // 2
                right, bottom = left + new_w, top + new_h
                
                # 在图像上标记放大区域
                from PIL import ImageDraw
                draw = ImageDraw.Draw(orig_img)
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                zoom_regions.append(orig_img)
    
    # 将提示词列表转换为Dataframe可接受的格式
    prompts_data = [[f"步骤 {i}", prompt] for i, prompt in enumerate(prompts)]
    
    # 返回处理结果前，主动清理显存
    gc.collect()
    torch.cuda.empty_cache()
    # 返回处理结果
    return results, concat, prompts_data, size_info, concat_path, zoom_regions

# Gradio界面
def create_ui():
    with gr.Blocks(title="Chain-of-Zoom 图像超分辨率工具", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Chain-of-Zoom 图像超分辨率工具")
        gr.Markdown("使用AI技术逐步放大图像细节，实现高质量的图像超分辨率")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="输入图像", type="pil")
                
                with gr.Group():
                    gr.Markdown("### 基本参数")
                    with gr.Row():
                        process_size = gr.Slider(label="处理尺寸", minimum=256, maximum=1024, value=512, step=64)
                        upscale = gr.Slider(label="每次放大倍数", minimum=2, maximum=8, value=4, step=1)
                    
                    with gr.Row():
                        rec_type = gr.Dropdown(
                            label="递归类型", 
                            choices=["recursive_multiscale", "recursive", "onestep", "nearest", "bicubic"], 
                            value="recursive_multiscale",
                            info="multiscale模式可获得最佳质量"
                        )
                        rec_num = gr.Slider(label="递归次数", minimum=1, maximum=6, value=4, step=1)
                
                with gr.Group():
                    gr.Markdown("### 高级参数")
                    with gr.Row():
                        prompt_type = gr.Dropdown(
                            label="提示词类型", 
                            choices=["vlm", "dape", "null"], 
                            value="vlm",
                            info="VLM使用视觉语言模型生成提示词"
                        )
                        align_method = gr.Dropdown(
                            label="颜色对齐方法", 
                            choices=["nofix", "wavelet", "adain"], 
                            value="nofix",
                            info="颜色修正算法"
                        )
                    
                    user_prompt = gr.Textbox(
                        label="自定义提示词（可选）", 
                        placeholder="输入额外的提示词，用于引导图像生成",
                        info="额外的提示词将与自动生成的提示词合并"
                    )
                    
                    with gr.Row():
                        efficient_memory = gr.Checkbox(label="高效内存模式", value=True, info="优化GPU内存使用")
                        mixed_precision = gr.Dropdown(label="精度", choices=["fp16", "fp32"], value="fp16", info="fp16速度更快，fp32质量更好")
                
                with gr.Group():
                    gr.Markdown("### 输出选项")
                    concat_layout = gr.Radio(
                        label="拼接布局", 
                        choices=["horizontal", "vertical", "grid"], 
                        value="horizontal",
                        info="选择最终拼接图的布局方式"
                    )
                
                with gr.Group():
                    gr.Markdown("### 提示词后端设置")
                    prompt_backend = gr.Radio(
                        label="提示词生成后端", 
                        choices=["local", "litellm"], 
                        value="local",
                        info="选择本地VLM或远程LiteLLM API"
                    )
                    litellm_api_base = gr.Textbox(label="LiteLLM API Base", placeholder="如 https://your-litellm-server.com")
                    litellm_api_key = gr.Textbox(label="LiteLLM API Key", type="password")
                    litellm_model = gr.Textbox(label="LiteLLM模型名", value="gpt-4")
                
                process_btn = gr.Button("开始处理", variant="primary")
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("结果展示"):
                        size_info = gr.Textbox(label="最终图像尺寸", interactive=False)
                        final_output = gr.Image(label="最终拼接结果", type="pil", interactive=False)
                        download_btn = gr.Button("下载完整拼接图", variant="secondary")
                        output_path = gr.Textbox(visible=False)  # 隐藏路径
                        
                        with gr.Accordion("所有步骤结果", open=False):
                            with gr.Row():
                                output_gallery = gr.Gallery(
                                    label="放大过程", 
                                    show_label=True,
                                    columns=5,
                                    object_fit="contain",
                                    height=300
                                )
                                # 修复：移除height参数
                                prompts_list = gr.Dataframe(
                                    headers=["步骤", "提示词"],
                                    datatype=["str", "str"],
                                    col_count=(2, "fixed"),
                                    interactive=False
                                )
                    
                    with gr.TabItem("放大区域可视化"):
                        zoom_regions_gallery = gr.Gallery(
                            label="每步放大区域", 
                            show_label=True,
                            columns=3,
                            object_fit="contain",
                            height=500
                        )
                
        # 处理函数连接
        process_outputs = process_btn.click(
            fn=process_image,
            inputs=[
                input_image, process_size, upscale, rec_type, rec_num,
                prompt_type, align_method, user_prompt, efficient_memory, mixed_precision,
                concat_layout, prompt_backend, litellm_api_base, litellm_api_key, litellm_model
            ],
            outputs=[
                output_gallery, final_output, prompts_list, size_info, output_path, zoom_regions_gallery
            ]
        )
        
        # 下载按钮功能
        def create_download_link(output_path):
            if not output_path:
                return None
            return output_path
            
        download_btn.click(
            fn=create_download_link,
            inputs=[output_path],
            outputs=[gr.File(label="下载")]
        )
        
        # 示例图像
        gr.Examples(
            examples=[
                "samples/0393.png",
                "samples/0457.png",
                "samples/0479.png"
            ],
            inputs=input_image,
            outputs=[output_gallery, final_output, prompts_list, size_info, output_path, zoom_regions_gallery],
            fn=process_image,
            cache_examples=False,
        )
        
        gr.Markdown("""
        ### 使用说明
        1. 上传一张图像或使用示例图像
        2. 调整参数（处理尺寸、放大倍数、递归次数等）
        3. 点击"开始处理"按钮
        4. 在结果标签页查看处理结果
        5. 使用"下载完整拼接图"按钮保存结果
        
        ### 参数解释
        - **处理尺寸**：输入图像的处理尺寸，更大的尺寸需要更多内存
        - **每次放大倍数**：每步放大的倍数，通常为4倍
        - **递归类型**：
          - recursive_multiscale: 多尺度递归放大（最佳质量）
          - recursive: 简单递归放大
          - onestep: 一次性放大
          - nearest/bicubic: 使用简单插值算法
        - **递归次数**：执行放大的次数，更多次数可以放大更小的细节
        - **提示词类型**：
          - vlm: 使用视觉语言模型生成提示词（推荐）
          - dape: 使用DAPE模型生成提示词
          - null: 不使用提示词
        - **颜色对齐**：对生成图像的颜色进行修正
        """)
        
    return demo

# 启动服务
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-backend', type=str, default='local', choices=['local', 'litellm'], help='提示词生成后端')
    parser.add_argument('--litellm-api-base', type=str, default='', help='LiteLLM API Base')
    parser.add_argument('--litellm-api-key', type=str, default='', help='LiteLLM API Key')
    parser.add_argument('--litellm-model', type=str, default='sonnet-3.7', help='LiteLLM模型名')
    args_cli = parser.parse_args()

    def create_ui_with_args():
        with gr.Blocks(title="Chain-of-Zoom 图像超分辨率工具", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🔍 Chain-of-Zoom 图像超分辨率工具")
            gr.Markdown("使用AI技术逐步放大图像细节，实现高质量的图像超分辨率")
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="输入图像", type="pil")
                    with gr.Group():
                        gr.Markdown("### 基本参数")
                        with gr.Row():
                            process_size = gr.Slider(label="处理尺寸", minimum=256, maximum=1024, value=512, step=64)
                            upscale = gr.Slider(label="每次放大倍数", minimum=2, maximum=8, value=4, step=1)
                        with gr.Row():
                            rec_type = gr.Dropdown(
                                label="递归类型", 
                                choices=["recursive_multiscale", "recursive", "onestep", "nearest", "bicubic"], 
                                value="recursive_multiscale",
                                info="multiscale模式可获得最佳质量"
                            )
                            rec_num = gr.Slider(label="递归次数", minimum=1, maximum=6, value=4, step=1)
                    with gr.Group():
                        gr.Markdown("### 高级参数")
                        with gr.Row():
                            prompt_type = gr.Dropdown(
                                label="提示词类型", 
                                choices=["vlm", "dape", "null"], 
                                value="vlm",
                                info="VLM使用视觉语言模型生成提示词"
                            )
                            align_method = gr.Dropdown(
                                label="颜色对齐方法", 
                                choices=["nofix", "wavelet", "adain"], 
                                value="nofix",
                                info="颜色修正算法"
                            )
                        user_prompt = gr.Textbox(
                            label="自定义提示词（可选）", 
                            placeholder="输入额外的提示词，用于引导图像生成",
                            info="额外的提示词将与自动生成的提示词合并"
                        )
                        with gr.Row():
                            efficient_memory = gr.Checkbox(label="高效内存模式", value=True, info="优化GPU内存使用")
                            mixed_precision = gr.Dropdown(label="精度", choices=["fp16", "fp32"], value="fp16", info="fp16速度更快，fp32质量更好")
                    with gr.Group():
                        gr.Markdown("### 输出选项")
                        concat_layout = gr.Radio(
                            label="拼接布局", 
                            choices=["horizontal", "vertical", "grid"], 
                            value="horizontal",
                            info="选择最终拼接图的布局方式"
                        )
                    with gr.Group():
                        gr.Markdown("### 提示词后端设置")
                        prompt_backend = gr.Radio(
                            label="提示词生成后端", 
                            choices=["local", "litellm"], 
                            value=args_cli.prompt_backend,
                            info="选择本地VLM或远程LiteLLM API"
                        )
                        litellm_api_base = gr.Textbox(label="LiteLLM API Base", value=args_cli.litellm_api_base, placeholder="如 https://your-litellm-server.com")
                        litellm_api_key = gr.Textbox(label="LiteLLM API Key", value=args_cli.litellm_api_key, type="password")
                        litellm_model = gr.Textbox(label="LiteLLM模型名", value=args_cli.litellm_model)
                    process_btn = gr.Button("开始处理", variant="primary")
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("结果展示"):
                            size_info = gr.Textbox(label="最终图像尺寸", interactive=False)
                            final_output = gr.Image(label="最终拼接结果", type="pil", interactive=False)
                            download_btn = gr.Button("下载完整拼接图", variant="secondary")
                            output_path = gr.Textbox(visible=False)
                            with gr.Accordion("所有步骤结果", open=False):
                                with gr.Row():
                                    output_gallery = gr.Gallery(
                                        label="放大过程", 
                                        show_label=True,
                                        columns=5,
                                        object_fit="contain",
                                        height=300
                                    )
                                    prompts_list = gr.Dataframe(
                                        headers=["步骤", "提示词"],
                                        datatype=["str", "str"],
                                        col_count=(2, "fixed"),
                                        interactive=False
                                    )
                        with gr.TabItem("放大区域可视化"):
                            zoom_regions_gallery = gr.Gallery(
                                label="每步放大区域", 
                                show_label=True,
                                columns=3,
                                object_fit="contain",
                                height=500
                            )
                process_outputs = process_btn.click(
                    fn=process_image,
                    inputs=[
                        input_image, process_size, upscale, rec_type, rec_num,
                        prompt_type, align_method, user_prompt, efficient_memory, mixed_precision,
                        concat_layout, prompt_backend, litellm_api_base, litellm_api_key, litellm_model
                    ],
                    outputs=[
                        output_gallery, final_output, prompts_list, size_info, output_path, zoom_regions_gallery
                    ]
                )
        return demo
    ui = create_ui_with_args()
    ui.launch(share=False, server_name="0.0.0.0")