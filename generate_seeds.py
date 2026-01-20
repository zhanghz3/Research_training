import argparse
import os
import json
import torch
from diffusers import FluxPipeline
from FLUXapi import ImageRequest
from evaluation.Evaluation import caption_score, vqa_score
from evaluation.VLManswer import model
import shutil
import requests
from tqdm import trange
import time
import openai
from evaluation.model import capture
from FLUXapi import ImageRequest

def get_claude_response(user_message):
    Baseurl = "https://api.claude-Plus.top"
    Skey = "sk-vjulMaFmBWm31NP4OqwnKaDJMb3X0jbVlnIvg4XbYgtXwzWi"

    payload = json.dumps({
        "model": "claude-3-5-sonnet-20240620",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    })

    url = Baseurl + "/v1/chat/completions"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {Skey}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload)

    data = response.json()

    content = data['choices'][0]['message']['content']

    return content

def get_gpt_response(content):
    client = openai.OpenAI(
        api_key="sk-QtJEJtma2OeZfonJd550SKxkjN07dG1H5LqyDrRC9gfQNN5B",
        base_url="https://api.xiaoai.plus/v1"
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
    )
    return completion.choices[0].message.content


def create_seed_json(code_list, objective_list, output_file):
    # 确保两个列表的长度相同
    if len(code_list) != len(objective_list):
        raise ValueError("code_list和objective_list的长度必须相同")

    # 构建JSON数据结构
    seed_data = []
    for code, objective in zip(code_list, objective_list): 
        entry = {
            "code": code,
            "objective": objective
        }
        seed_data.append(entry)
        print("code",code,"objective", objective)

    # 将数据写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(seed_data, file, ensure_ascii=False, indent=4)


def check_response(response):
    if not response.startswith("A"):
        # 找到第一个换行符的位置
        newline_index = response.find('\n')
        # 如果找到了换行符，就删除从开头到换行符之间的内容
        if newline_index != -1:
            return response[newline_index + 1:] + '\n'  # +1是为了包含换行符
    else:
        return response + '\n'


def initialize(model_name, model, processor, pattern, evaluator):
    while True:
        try:
            generated_prompts1 = get_claude_response(prompt_animal)
            generated_prompts1 = check_response(generated_prompts1)
            print(generated_prompts1)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    time.sleep(60)
    while True:
        try:
            generated_prompts2 = get_gpt_response(prompt_animal)
            generated_prompts2 = check_response(generated_prompts2)
            print(generated_prompts2)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    time.sleep(60)
    while True:
        try:
            generated_prompts3 = get_claude_response(prompt_people)
            generated_prompts3 = check_response(generated_prompts3)
            print(generated_prompts3)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    time.sleep(60)
    while True:
        try:
            generated_prompts4 = get_gpt_response(prompt_people)
            generated_prompts4 = check_response(generated_prompts4)
            print(generated_prompts4)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    time.sleep(60)
    while True:
        try:
            generated_prompts5 = get_claude_response(prompt_object)
            generated_prompts5 = check_response(generated_prompts5)
            print(generated_prompts5)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    time.sleep(60)
    while True:
        try:
            generated_prompts6 = get_gpt_response(prompt_object)
            generated_prompts6 = check_response(generated_prompts6)
            print(generated_prompts6)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    generated_prompts = generated_prompts1 + generated_prompts2 + generated_prompts3 + generated_prompts5 + generated_prompts6
    generated_prompts_list = [prompt.strip() for prompt in generated_prompts.split('\n') if prompt.strip()]
    code_list.extend(generated_prompts_list)
    print(f"=============total generate {len(code_list)} prompts ============")


    with open(args.captionfilename, "w", encoding="utf-8") as file:
        for i in trange(len(code_list)):
            caption = code_list[i]
            print(caption)
            # 打开文件并写入字符串
            file.write(caption + '\n')  # 写入字符串并在每个字符串后添加换行符
            print("已经写入")
            '''
            image = pipe(
                caption,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
            '''
            # this will create an api request directly but not block until the generation is finished
            request = ImageRequest(caption, name="flux.1-dev",api_key="a69d4119-3507-4cc8-b6a3-22e5d961c5c9")
            # or: request = ImageRequest("A beautiful beach", name="flux.1.1-pro", api_key="your_key_here")
            image_path = os.path.join(args.imagefilename, f"{i}.jpg")
            request.save(image_path)
            if pattern == 'caption':
                fitness,_ = caption_score(model_name, model, processor, image_path, caption, evaluator)
            elif pattern == 'VQA':
                fitness,_ = vqa_score(model_name, model, processor, image_path, caption, evaluator)
            print(fitness)
            objective_list.append(fitness)

if __name__ == '__main__':      
    torch.multiprocessing.set_start_method("spawn")  
    #Adding necessary input arguments
    parser = argparse.ArgumentParser(description='generate_seeds_captions')
    parser.add_argument('--captionfilename', type=str)
    parser.add_argument('--imagefilename', type=str)
    parser.add_argument('--output_file',default='ael_seeds/seeds.json')
    # 使用示例
    # model_path = "model/llava-v1.5-7b"
    # model_path = "model/llava-v1.5-13b"
    # model_path = "model/llava-v1.6-vicuna-7b"
    # model_path = "model/llava-v1.6-vicuna-13b"
    parser.add_argument('--model_name', default = "gpt-4o", type=str)
    parser.add_argument('--pattern', default="caption", type=str)


    args = parser.parse_args()
    code_list = []
    objective_list = []
    if os.path.exists(args.captionfilename):
        os.remove(args.captionfilename)
    if os.path.exists(args.imagefilename):
        shutil.rmtree(args.imagefilename)
        os.mkdir(args.imagefilename)
    else:
        os.mkdir(args.imagefilename)
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    prompt = "Requirements: Generate 100 diverse image description sentences according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <number> <color> <object> <appearance> in the style of <style>. <It/He/She/They> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. For example, <object> can be filled with any person, object, animal, building, etc. Do not be limited to a certain category and try to increase the diversity and imagination as much as possible. <appearance> can be filled in with appearance descriptions such as wearing glasses. Example: a picture of a blue dog wearing sunglasses in the style of realistic. It is sitting on the beach in the moon on a snowy day, it is drinking a bottle of cola. There are many medieval castles around and many spaceships in the sky."
    prompt_people = "Requirements: Generate 100 diverse image description sentences according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <number> <color> <people> <appearance> in the style of <style>. <He/She/They> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. For example, <people> can be only filled with any human and humanoid concepts of any race, age, gender, appearance, identity, and shape. Try to increase the diversity and imagination as much as possible. <appearance> can be filled in with appearance descriptions such as wearing glasses. Example: a picture of a white man wearing sunglasses in the style of realistic. It is sitting on the beach in the moon on a snowy day, it is drinking a bottle of cola. There are many medieval castles around and many spaceships in the sky."
    prompt_object = "Requirements: Generate 100 diverse image description sentences according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <number> <color> <thing> <appearance> in the style of <style>. <It/They> <is/are> <on/at/in> the <background> in the <location> on a <weather> day, <Detailed description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. For example, <thing> can be only filled with any environments, buildings, things, transportations, food, treasure, unknown objects but not living creatures. Do not be limited to a certain category and Try to increase the diversity as much as possible. <appearance> can be filled in with appearance descriptions. Example: a picture of a football with flowers blooming on top in the style of realistic. It is at the morden city in the moon on a snowy day, it has hieroglyphs written on its surface and has beautiful classical patterns. There are many medieval castles around and many spaceships in the sky."
    prompt_animal = "Requirements: Generate 100 diverse image description sentences according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <number> <color> <animal> <appearance> in the style of <style>. <It/They> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. For example, <animal> can be only filled with animals. Do not be limited to a certain category and try to increase the diversity and imagination as much as possible. <appearance> can be filled in with appearance descriptions such as wearing glasses. Example: a picture of a blue dog wearing sunglasses in the style of realistic. It is sitting on the beach in the moon on a snowy day, it is drinking a bottle of cola. There are many medieval castles around and many spaceships in the sky."
    '''
    print(">>> Loading FLUX....")
    pipe = FluxPipeline.from_pretrained("/home/pod/shared-nvme/models/FLUX.1-dev", torch_dtype=torch.bfloat16, device_map='balanced')
    '''
    print(">>> Loading CAPTURE....")
    evaluator = capture.CAPTURE()
    print(f"当前设置:\ncaption->{args.captionfilename}\nimage->{args.imagefilename}\npattern->{args.pattern}\noutput_file->{args.output_file}")
    if args.model_name == "gpt-4o":
        initialize("gpt-4o",None, None, args.pattern, evaluator)
    else:
        Model, Processor = model(args.model_name)
        initialize(args.model_name, Model, Processor, args.pattern, evaluator)

    # 调用函数生成seed.json文件
    create_seed_json(code_list, objective_list, args.output_file)
    print('generate seeds success!')



