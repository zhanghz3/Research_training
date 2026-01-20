import time
import torch
import json
from PIL import Image
from evaluation.Evaluation import caption_score,vqa_score
from FLUXapi import ImageRequest
import random

def FLUX_api(caption, image_path):
    api_keys = ["13dacd96-d758-472d-9bb9-735e4543c52b","a69d4119-3507-4cc8-b6a3-22e5d961c5c9", "26867829-97b3-4656-8c9c-8a396083eb69", "347829b4-eed8-472a-9f33-cd8788dcfbab"]
    while True:
        try:
            api_key = random.choice(api_keys)
            request = ImageRequest(caption, name="flux.1-dev",api_key=api_key)
            request.save(image_path)
            print("Image saved successfully.")
            break  # 成功后退出循环
        except Exception as e:
            print(f"Failed to save image, retrying...{e}")
            time.sleep(3)
            

class Evaluation:
    def __init__(self, localT2I,T2Imodel, model_name, model, processor, pattern, evaluator):
        print("begin evaluate")
        self.stages = {}
        self.index = 0
        self.pattern = pattern
        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.evaluator = evaluator
        self.localT2I = localT2I
        if self.localT2I:
            self.T2I = T2Imodel
    def evaluate(self):
        try:
            with open("caption.txt", "r", encoding="utf-8") as file:
                caption = file.read()
            if self.pattern == 'caption':
                score,vlmanswer = self.get_score(caption)
                return score,vlmanswer
            elif self.pattern == 'VQA':
                score = self.get_score(caption)
            return score
        except Exception as e:
            print("Error:",str(e))
            return None

    def get_score(self, caption):
        print(caption)
        if self.localT2I:
            image = self.T2I(
                caption,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
            image_path = f"./images_{self.model_name}_{self.pattern}/{self.index}.jpg"
            image.save(image_path)
            self.stages[caption] = image_path
            self.index = self.index + 1
        else:
            image_path = f"./images_{self.model_name}_{self.pattern}/{self.index}.jpg"
            FLUX_api(caption,image_path)
            self.stages[caption] = image_path
            self.index = self.index + 1
        if self.pattern == 'caption':
            fitness, answer = caption_score(self.model_name, self.model, self.processor, image_path, caption, self.evaluator)
            return fitness, answer
        elif self.pattern == 'VQA':
            fitness, answer = vqa_score(self.model_name, self.model, self.processor, image_path, caption, self.evaluator)
            return fitness, answer


class GetPrompts:
    def __init__(self):
        self.prompt_task = "You need to change the caption in the equation."
        self.prompt_func_name = "captionUpdate"
        self.prompt_func_inputs = [""]
        self.prompt_func_outputs = ["caption"]
        self.prompt_inout_inf = "You only need to change the content of the caption in the method."

        self.prompt_other_inf = "return caption:The changed value"

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf

