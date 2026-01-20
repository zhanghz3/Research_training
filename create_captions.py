import openai
import httpx
import time
from tqdm import trange
from FLUXapi import ImageRequest
import random

'''

def check_response(response):
    if not response.startswith("A"):
        # 找到第一个换行符的位置
        newline_index = response.find('\n')
        # 如果找到了换行符，就删除从开头到换行符之间的内容
        if newline_index != -1:
            return response[newline_index + 1:] + '\n'  # +1是为了包含换行符
    else:
        return response + '\n'

def get_claude_response(user_message):
    print("现在是Claude")
    client = openai.OpenAI(
    base_url="https://xiaoai.plus/v1",
    #base_url="https://api.aikeji.vip/v1",
    api_key="sk-UnPsBuQhdwx6Xok3nMU44MeBkZL0UmBAZQuyeeJ40Ra6N4N6",
    #api_key="sk-OFFKKijtNTHC0kLAyWB8ji5XYIeKJ9letJuYzuw7sVYR3qgd",
    http_client=httpx.Client(
        base_url="https://xiaoai.plus/v1",
        #base_url="https://api.aikeji.vip/v1",
        follow_redirects=True,
        ),
    )

    completion = client.chat.completions.create(
    model="claude-3-5-sonnet-20240620",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
    temperature = 0.9
    )
    return completion.choices[0].message.content

def get_gpt_response(content):
    print("现在是GPT")
    client = openai.OpenAI(
        api_key="sk-OiW3PhgMaGG0JsiVd2XCCPQE0T6jj2EEMp93o5lx7Iao1hbc",
        base_url="https://api.xiaoai.plus/v1"
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        temperature = 0.9
    )
    return completion.choices[0].message.content

def initialize():
    while True:
        try:
            generated_prompts1 = get_claude_response(prompt_people)
            generated_prompts1 = check_response(generated_prompts1)
            print(generated_prompts1)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts2 = get_gpt_response(prompt_people)
            generated_prompts2 = check_response(generated_prompts2)
            print(generated_prompts2)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts3 = get_claude_response(prompt_people_i)
            generated_prompts3 = check_response(generated_prompts3)
            print(generated_prompts3)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts4 = get_gpt_response(prompt_people_i)
            generated_prompts4 = check_response(generated_prompts4)
            print(generated_prompts4)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts5 = get_claude_response(prompt_object)
            generated_prompts5 = check_response(generated_prompts5)
            print(generated_prompts5)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts6 = get_gpt_response(prompt_object)
            generated_prompts6 = check_response(generated_prompts6)
            print(generated_prompts6)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts7 = get_claude_response(prompt_object_i)
            generated_prompts7 = check_response(generated_prompts7)
            print(generated_prompts7)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts8 = get_gpt_response(prompt_object_i)
            generated_prompts8 = check_response(generated_prompts8)
            print(generated_prompts8)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts9 = get_claude_response(prompt_animal)
            generated_prompts9 = check_response(generated_prompts9)
            print(generated_prompts9)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts10 = get_gpt_response(prompt_animal)
            generated_prompts10 = check_response(generated_prompts10)
            print(generated_prompts10)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts11 = get_claude_response(prompt_animal_i)
            generated_prompts11 = check_response(generated_prompts11)
            print(generated_prompts11)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts12 = get_gpt_response(prompt_animal_i)
            generated_prompts12 = check_response(generated_prompts12)
            print(generated_prompts12)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts13 = get_claude_response(prompt_environment)
            generated_prompts13 = check_response(generated_prompts13)
            print(generated_prompts13)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts14 = get_gpt_response(prompt_environment)
            generated_prompts14 = check_response(generated_prompts14)
            print(generated_prompts14)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts15 = get_claude_response(prompt_environment_i)
            generated_prompts15 = check_response(generated_prompts15)
            print(generated_prompts15)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    while True:
        try:
            generated_prompts16 = get_gpt_response(prompt_environment_i)
            generated_prompts16 = check_response(generated_prompts16)
            print(generated_prompts16)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)  # 休眠10秒后重试
    generated_prompts = generated_prompts1 + generated_prompts2 + generated_prompts3 + generated_prompts4 + generated_prompts5 + generated_prompts6 + generated_prompts7 + generated_prompts8 + generated_prompts9 + generated_prompts10 + generated_prompts11 + generated_prompts12 + generated_prompts13 + generated_prompts14 + generated_prompts15 + generated_prompts16
    generated_prompts_list = [prompt.strip() for prompt in generated_prompts.split('\n') if prompt.strip()]
    code_list.extend(generated_prompts_list)
    print(f"=============total generate {len(code_list)} prompts ============")


    with open("./caption.txt", "w", encoding="utf-8") as file:
        for i in trange(len(code_list)):
            caption = code_list[i]
            print(caption)
            # 打开文件并写入字符串
            file.write(caption + '\n')  # 写入字符串并在每个字符串后添加换行符
            print("已经写入")
'''
            
def FLUX_api(caption, image_path):
    api_keys = ["13dacd96-d758-472d-9bb9-735e4543c52b", "a69d4119-3507-4cc8-b6a3-22e5d961c5c9", "26867829-97b3-4656-8c9c-8a396083eb69", "347829b4-eed8-472a-9f33-cd8788dcfbab"]
    failure_count = 0  # 初始化失败次数计数器
    while failure_count < 5:  # 如果失败次数超过5次，则跳过
        try:
            api_key = random.choice(api_keys)
            # 假设ImageRequest和save方法已经定义
            request = ImageRequest(caption, name="flux.1-dev", api_key=api_key)
            request.save(image_path)
            print("Image saved successfully.")
            break  # 成功后退出循环
        except Exception as e:
            print(f"Failed to save image, retrying...{e}")
            time.sleep(10)
            failure_count += 1  # 失败次数加1
    if failure_count >= 5:  # 如果失败次数达到5次，从列表和文件中删除该code
        print(f"Skipping code: {caption} due to excessive failures.")
        lines_list.remove(caption)
        with open('caption.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
        with open('caption.txt', 'w', encoding='utf-8') as file:
            for line in lines:
                if line.strip() != caption:
                    file.write(line)

'''
prompt_people = "Requirements: Generate 100 diverse image description sentences about human according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <number> <color> <people> <appearance> in the style of <style>. <He/She/They> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. <people> can be only filled with any human of any race, age, gender, appearance, identity, and shape. <appearance> can be filled in with appearance descriptions such as wearing glasses. Example: a picture of a white man wearing sunglasses in the style of realistic. He is sitting on the beach in the house on a snowy day, he is drinking a bottle of cola. There are many ships around."
prompt_people_i = "Requirements: Generate 100 diverse and imaginative image description sentences about human according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <number> <color> <people> <appearance> in the style of <style>. <He/She/They> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. <people> can be only filled with any human of any race, age, gender, appearance, identity, and shape. Try to increase the diversity as much as possible. <appearance> can be filled in with appearance descriptions such as wearing glasses. Try to increase the diversity and imagination as much as possible. Example: a picture of a white man wearing sunglasses in the style of realistic. He is sitting on the beach in the moon on a snowy day, he is drinking a bottle of cola. There are many medieval castles around and many spaceships in the sky."
prompt_object_i = "Requirements: Generate 100 diverse and imaginative image description sentences about objects and things according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <number> <color> <thing> <appearance> in the style of <style>. <It/They> <is/are> <on/at/in> the <background> in the <location> on a <weather> day, <Detailed description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. <thing> can be only filled with any objects, buildings, things, transportations, food, treasure, unknown objects but not living creatures. Do not be limited to a certain category and Try to increase the diversity and imagination as much as possible. <appearance> can be filled in with appearance descriptions. Example: a picture of a football with flowers blooming on top in the style of realistic. It is at the morden city in the moon on a snowy day, it has hieroglyphs written on its surface and has beautiful classical patterns. There are many medieval castles around and many spaceships in the sky."
prompt_object = "Requirements: Generate 100 diverse image description sentences about objects and things according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <number> <color> <thing> <appearance> in the style of <style>. <It/They> <is/are> <on/at/in> the <background> in the <location> on a <weather> day, <Detailed description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. <thing> can be only filled with any objects, buildings, things, transportations, food, treasure, but not living creatures. <appearance> can be filled in with appearance descriptions. Example: a picture of a football in the style of realistic. It is at the morden city in the grass on a sunny day, it has beautiful patterns. There are many buildings around."
prompt_animal_i = "Requirements: Generate 100 diverse image description sentences about animals according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <number> <color> <animal> <appearance> in the style of <style>. <It/They> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. <animal> can be only filled with animals. Do not be limited to a certain category and try to increase the diversity and imagination as much as possible. <appearance> can be filled in with appearance descriptions such as wearing glasses. Example: a picture of a blue dog wearing sunglasses in the style of realistic. It is sitting on the beach in the moon on a snowy day, it is drinking a bottle of cola. There are many medieval castles around and many spaceships in the sky."
prompt_animal = "Requirements: Generate 100 diverse image description sentences about animals according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <number> <color> <animal> <appearance> in the style of <style>. <It/They> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. <animal> can be only filled with animals. <appearance> can be filled in with appearance descriptions."
prompt_environment = "Requirement: Generate 100 diverse picture description sentences about environments, scenes and scenery according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <color> <environments/scenes/scenery> <appearance> in the style of <style>. <It/They> <is/are> <on/at/in> the <background> in the <location> on a <weather> day, <Detailed description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. <environments/scenes/scenery> can be only filled with any environments, scenes and scenery. <appearance> can be filled in with appearance descriptions. "
prompt_environment_i = "Requirement: Generate 100 diverse and imaginative picture description sentences about environments, scenes and scenery according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <color> <environments/scenes/scenery> <appearance> in the style of <style>. <It/They> <is/are> <on/at/in> the <background> in the <location> on a <weather> day, <Detailed description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. <environments/scenes/scenery> can be only filled with any environments, scenes and scenery. Do not be limited to a certain category and try to increase the diversity and imagination as much as possible. <appearance> can be filled in with appearance descriptions."

code_list = []
initialize()
print("种子已经写入caption")
print(f"种子数量为{len(code_list)}")
'''
# 初始化一个空列表来存储行内容
lines_list = []

# 打开文件并读取每一行
with open('caption.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # 将每一行的末尾的换行符去除后添加到列表中
        lines_list.append(line.strip())

# 对每个code调用FLUX_api函数
for i, code in enumerate(lines_list):
    FLUX_api(code, f"./image/{i+1}.jpg")