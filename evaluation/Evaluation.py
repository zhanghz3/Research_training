import re
from evaluation.VLManswer import model, get_response
import numpy as np


def check_response(response):
    # 检查是否包含非法标点符号
    if re.search(r'[<>/]', response):
        return False

    # 如果所有检查都通过
    return True

def caption_score(model_name, model, processor, image_path, caption, evaluator):
    prompt = "Please describe the image in the following format: a <picture/photo/watercolor/sketch> of <number> <color> <object> <appearance> in the style of <style>. <They/It/He/She> <gesture> on the <background describe> in the <location> on a <weather> day, <action description>, <environment description>. Do not add extraneous statements and strictly follow the instructions."
    answer = get_response(model_name, model, processor, image_path, prompt)
    caption = {'prompt':[caption]}
    answer = {'prompt':[answer]}
    score = evaluator.compute_score(caption, answer)[0]
    # 检查返回值类型，并相应地处理
    if isinstance(score, (np.ndarray, np.generic)):
        # 如果是numpy数组或numpy的标量类型，直接返回其值
        return score.item(), answer['prompt'][0]
    elif isinstance(score, float):
        # 如果是Python的原生float类型，也直接返回
        return score, answer['prompt'][0]
    else:
        raise TypeError("Unsupported type returned from compute_score: {}".format(type(score)))
    
def vqa_score(model_name, model, processor, image_path, caption, evaluator):
    prompt = "Please answer my following questions in detail based on the content of the picture. You need to write your answer as a continuous descriptive sentence. What type of picture is this? For example: a photo, sketch, watercolor painting, or another type? What color should the main object(s) be? What is the main object in this image? Are there any specific appearance features? What style would you like? For example: vintage, modern illustration, or a specific artist's style? What gesture or action is the object doing in the image? What type of background setting is this? Are there specific scene elements or details? Where is this scene set? What is the weather like in the scene? Are there any specific actions or dynamics in the scene? For instance: swaying in the wind, interacting with another object, etc. How would you describe the overall atmosphere of the surroundings? "
    answer = get_response(model_name, model, processor, image_path, prompt)
    caption = {'prompt':[caption]}
    answer = {'prompt':[answer]}
    score = evaluator.compute_score(caption, answer)[0]
    # 检查返回值类型，并相应地处理
    if isinstance(score, (np.ndarray, np.generic)):
        # 如果是numpy数组或numpy的标量类型，直接返回其值
        return score.item(), answer['prompt'][0]
    elif isinstance(score, float):
        # 如果是Python的原生float类型，也直接返回
        return score, answer['prompt'][0]
    else:
        raise TypeError("Unsupported type returned from compute_score: {}".format(type(score)))




