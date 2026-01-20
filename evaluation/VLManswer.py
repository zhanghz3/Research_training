from openai import OpenAI
import re
import requests
import time
import base64
import math
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
import numpy as np

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def check_response(response):
    # 检查是否包含非法标点符号
    if re.search(r'[<>/]', response):
        return False

    # 如果所有检查都通过
    return True

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def model(model_name):
    if model_name == "InternVL":
        path = "/home/pod/shared-nvme/models/InternVL2-8B"
        device_map = split_model('InternVL2-8B')
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            device_map=device_map).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
        return model, tokenizer
    elif model_name == "llama":
        model = MllamaForConditionalGeneration.from_pretrained(
                    '/home/pod/shared-nvme/models/Llama-3.2-11B-Vision-Instruct',
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
        processor = AutoProcessor.from_pretrained('/home/pod/shared-nvme/models/Llama-3.2-11B-Vision-Instruct')
        return model, processor
    elif model_name == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(
                    '/home/pod/shared-nvme/models/llava-1.5-13b-hf', 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True, 
                    device_map = 'auto'
                )
        processor = AutoProcessor.from_pretrained('/home/pod/shared-nvme/models/llava-1.5-13b-hf')
        return model, processor
    elif model_name == "llava-next":
        processor = LlavaNextProcessor.from_pretrained("/home/pod/shared-nvme/models/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained("/home/pod/shared-nvme/models/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto') 
        return model, processor
    elif model_name == "Molmo":
        # load the processor
        processor = AutoProcessor.from_pretrained(
            '/home/pod/shared-nvme/models/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            '/home/pod/shared-nvme/models/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        model.to(dtype=torch.bfloat16)
        return model, processor
    elif model_name == "Qwen":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            '/home/pod/shared-nvme/models/通义千问2-VL-7B-Instruct', torch_dtype="auto", device_map="auto"
        )
        # default processer
        processor = AutoProcessor.from_pretrained('/home/pod/shared-nvme/models/通义千问2-VL-7B-Instruct')
        return model, processor
    else:
        raise Exception("没有这个模型！")

def model_api(image_path, question):
    base64_image = encode_image(image_path)
    #client = OpenAI(api_key="sk-b504E9yQbZnvNJUn9b02Fe3d313d420bA11761Ae3a11AfCa",
    #        base_url="https://api.gptapi.us/v1/chat/completions")
    key = ["sk-QtJEJtma2OeZfonJd550SKxkjN07dG1H5LqyDrRC9gfQNN5B","sk-M8cqwQpKCQFZM8bMwJ8XJ3c7mQ9vjhhYWtBUebgunwBaFAv4"]
    idx = 1
    while True:
        client = OpenAI(api_key=key[idx%2],
            base_url="https://api.xiaoai.plus/v1")
        try:
            completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                             "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    },
                ],
                }
            ],
            )
            content = completion.choices[0].message.content
            print("得到4v响应")
            if check_response(content):
                print("检查通过")
                return content
            else:
                continue
        except requests.exceptions.RequestException as e:
            print(f"请求失败，正在重试... ({e})")
            idx = idx+1
            print("换用另一个api")
            time.sleep(30)  # 等待30秒后重试
        except Exception as e:
            print(f"发生未知错误，正在重试... ({e})")
            idx = idx+1
            print("换用另一个api")
            time.sleep(30)  # 等待30秒后重试
            
def get_response(model_name, model, processor, image_path, query):
    if model_name == "gpt-4o":
        return model_api(image_path, query)
    if model_name == "InternVL":
        # set the max number of tiles in `max_num`
        pixel_values = load_image(image_path, max_num=1).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=512, do_sample=True)

        question = '<image>\n'+query
        response = model.chat(processor, pixel_values, question, generation_config)
        return response
    if model_name == "llama":
        image = Image.open(image_path)

        messages = [
            {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": query}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=512)
        text = processor.decode(output[0])
        pattern = r'<\|end_header_id\|>(.*?)<\|eot_id\|>.*?<\|end_header_id\|>(.*?)<\|eot_id\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
        # 检查是否有足够的匹配组
            if len(matches) > 1:
            # 返回第二组匹配的内容
                return matches[1][0]
            else:
            # 如果没有找到第二组匹配的内容，返回提示信息
                return text
        else:
            # 如果没有找到任何匹配的内容，返回提示信息
            return text
    if model_name == "llava":
        raw_image = Image.open(image_path)
        prompt = "USER: <image>\n"+query+"\nASSISTANT:"
        inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = processor.decode(output[0][2:], skip_special_tokens=True)
        start_index = text.find("ASSISTANT: ")
        if start_index != -1:
        # 截取"ASSISTANT: "之后的内容
            return text[start_index + len("ASSISTANT: "):]
        else:
        # 如果没有找到"ASSISTANT: "
            return text
    if model_name == "llava-next":
        # prepare image and text prompt, using the appropriate prompt template
        image = Image.open(image_path)
        prompt = "[INST] <image>\n"+query+" [/INST]"

        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt      
        output = model.generate(**inputs, max_new_tokens=512)
        text = processor.decode(output[0], skip_special_tokens=True)
        start_index = text.find("[/INST]")
        if start_index != -1:
            return text[start_index + len("[/INST]"):]
        else:
            return text
    if model_name == "Molmo":
        # process the image and text
        inputs = processor.process(
            images=[Image.open(image_path)],
            text=query
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        inputs["images"] = inputs["images"].to(torch.bfloat16)

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
    if model_name == "Qwen":
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    else:
        raise Exception("没有这个模型！")
    
if __name__ == '__main__':

    model_list = ['InternVL','llama','llava','llava-next','Molmo','Qwen']
    for model_name in model_list:
        print(f"现在是{model_name}")
        if model_name == 'gpt-4o':
            print(get_response(model_name, None, None, '/home/pod/shared-nvme/test/FLUXapi_test.jpg', '构思一个在这个场景中的故事'))
        else:
            Model, Processor = model(model_name)
            print(get_response(model_name, Model, Processor, '/home/pod/shared-nvme/test/FLUXapi_test.jpg', '构思一个在这个场景中的故事'))