import re
from openai import OpenAI
import httpx
import random

def check_response(response):
    # 检查是否包含非法标点符号
    if re.search(r'[<>/*]', response):
        return False

    # 如果所有检查都通过
    return True
'''
def get_response(content):
    key_dict = {"sk-QtJEJtma2OeZfonJd550SKxkjN07dG1H5LqyDrRC9gfQNN5B":"https://xiaoai.plus/v1",
            "sk-OFFKKijtNTHC0kLAyWB8ji5XYIeKJ9letJuYzuw7sVYR3qgd":"https://api.aikeji.vip/v1",
            "sk-UnPsBuQhdwx6Xok3nMU44MeBkZL0UmBAZQuyeeJ40Ra6N4N6":"https://xiaoai.plus/v1",
            "sk-OiW3PhgMaGG0JsiVd2XCCPQE0T6jj2EEMp93o5lx7Iao1hbc":"https://xiaoai.plus/v1"}
    key = ["sk-QtJEJtma2OeZfonJd550SKxkjN07dG1H5LqyDrRC9gfQNN5B","sk-OFFKKijtNTHC0kLAyWB8ji5XYIeKJ9letJuYzuw7sVYR3qgd","sk-UnPsBuQhdwx6Xok3nMU44MeBkZL0UmBAZQuyeeJ40Ra6N4N6","sk-OiW3PhgMaGG0JsiVd2XCCPQE0T6jj2EEMp93o5lx7Iao1hbc"]
    idx = 1
    while True:
        print(f"第{idx}次尝试")
        api_key = random.choice(key)
        base_url = key_dict[api_key]
        print("现在的api是",base_url)
        client = OpenAI(
            base_url=base_url,
            #base_url="https://api.aikeji.vip/v1",
            api_key=api_key,
            #api_key="sk-OFFKKijtNTHC0kLAyWB8ji5XYIeKJ9letJuYzuw7sVYR3qgd",
            http_client=httpx.Client(
                    base_url=base_url,
                    #base_url="https://api.aikeji.vip/v1",
                    follow_redirects=True,
            ),
        )
        try:
            print("发送请求")
            completion = client.chat.completions.create(
                    model="claude-3-5-sonnet-20240620",
                    messages=[
                        {"role": "system", "content": "You are a writer who is good at writing descriptive sentences, you have a wealth of knowledge and creativity, and the content you create is very diverse."},
                        {"role": "user", "content": content}
                    ],
                    temperature=1.0
            )
            response = completion.choices[0].message.content
            print("得到回应")
            if check_response(response):
                print("检查通过")
                return response
            else:
                print("检查不通过，重试")
                continue
        except Exception as e:
            print(f"错误发生：{e}")
            idx = idx + 1
            print("换用另一个api")

'''
def get_response(content):
    key_dict = {"sk-adslhgLa6XnelGerC485164223B14105A4A28206F8BeEaC2":"https://api.gptapi.us/v1/chat/completions",
            "sk-QtJEJtma2OeZfonJd550SKxkjN07dG1H5LqyDrRC9gfQNN5B":"https://api.xiaoai.plus/v1"}
    key = ["sk-adslhgLa6XnelGerC485164223B14105A4A28206F8BeEaC2","sk-QtJEJtma2OeZfonJd550SKxkjN07dG1H5LqyDrRC9gfQNN5B"]
    idx = 1
    while True:
        print(f"第{idx}次尝试")
        api_key = key[idx%2]
        base_url = key_dict[api_key]
        print("现在的api是",base_url)
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        try:
            print("发送请求")
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a writer who is good at writing descriptive sentences."},
                    {"role": "user", "content": content}
                ],
                temperature=0.9
            )
            response = completion.choices[0].message.content
            print("得到回应")
            if check_response(response):
                print("检查通过")
                return response
            else:
                print("检查不通过，重试")
                continue
        except Exception as e:
            print(f"错误发生：{e}")
            idx = idx + 1
            print("换用另一个api")

'''
for i in range(20):
    print(get_response("Requirements: Generate a diverse image description sentences according to the sentence format I give below, separeted by '\n'. Do not add any numbering, bullets or irrelevant texts, strictly follow the instructions. You can not repeat. Format: a <picture/photo/watercolor/sketch> of <number> <color> <object> <appearance> in the style of <style>. <It/He/She/They> <gesture> on the <background> in the <location> on a <weather> day, <action description>, <environment description>. Note: <picture/photo/watercolor/sketch> indicates that these are four options. When you generate a sentence, you can choose one of these four. The rest of the content in <...> specifies what kind of content you should fill in this position. For example, <object> can be filled with any person, object, animal, building, etc. Do not be limited to a certain category and try to increase the diversity as much as possible. <appearance> can be filled in with appearance descriptions such as wearing glasses. Example: a picture of a blue dog wearing sunglasses in the style of realistic. It is sitting on the beach in the moon on a snowy day, it is drinking a bottle of cola. There are many medieval castles around and many spaceships in the sky."))
'''