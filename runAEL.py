import argparse
import sys
import os
import json
import shutil
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(ABS_PATH, "..", "..")
sys.path.append(ROOT_PATH)  # This is for finding all the modules
from aell.src.aell import ael
from aell.src.aell.utils import createFolders
import torch
from diffusers import FluxPipeline
from prepareAEL import *
from evaluation.model import capture
from evaluation.VLManswer import model

def get_output(results, caption_path, image_path, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    code_list = [entry["code"] for entry in data.values()]
    os.makedirs(os.path.dirname(caption_path), exist_ok=True)

    with open(caption_path, 'w', encoding='utf-8') as file:
        for caption in code_list:
            file.write(caption + "\n")
            image = results[caption]
            shutil.copy(image, image_path)


if __name__ == '__main__':      
    torch.multiprocessing.set_start_method("spawn") 
    ### Debug model ###
    debug_mode = False# if debug

    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description='main')
    # number of algorithms in each population, default = 10
    parser.add_argument('--pop_size', default=20, type=int)#40
    # number of populations, default = 5
    parser.add_argument('--n_pop',default=5,type=int)
    # number of parents for 'e1' and 'e2' operators, default = 2
    parser.add_argument('--m',default=2,type=int)
    parser.add_argument('--pop_save_number', default=[20,15,10,10,5,5])#[20,20,20,15,10,5,4,3,2,0]
    parser.add_argument('--pop_size_epoch', default=[15,10,10,5,5])#[35,30,25,20,15,10,5,5,5]
    parser.add_argument('--n_pop_epoch', default=[5,5,5,5,5])#[3,3,3,3,3,3,3,3,3]
    parser.add_argument('--m_epoch', default=[2,2,2,2,2])#[2,2,2,2,2,2,2,2,2]
    # 使用示例
    # model_path = "model/llava-v1.5-7b"
    # model_path = "model/llava-v1.5-13b"
    # model_path = "model/llava-v1.6-vicuna-7b"
    # model_path = "model/llava-v1.6-vicuna-13b"
    parser.add_argument('--seedpath',type=str)
    parser.add_argument('--model_name', default="gpt-4o", type=str)
    parser.add_argument('--pattern', default="caption", type=str)
    parser.add_argument('--localT2I', default=False, type=bool)

    args = parser.parse_args()

    ### LLM settings  ###
    use_local_llm = False  # if use local model
    url = None  # your local server 'http://127.0.0.1:11012/completions'
    ### output path ###
    output_path = "./"  # default folder for ael outputs
    createFolders.create_folders(output_path, f"result/{args.model_name}_{args.pattern}")
    load_data = {
        'use_seed': True,
        'seed_path': output_path + args.seedpath,
        "use_pop": False,
        "pop_path": output_path + f"result/{args.model_name}_{args.pattern}" + "/pops/population_generation_0.json",
        "n_pop_initial": 0
    }
    ### Debug model ###
    debug_mode = False  # if debug

    # AEL
    operators = ['e1','m1']  # evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
    operator_weights = [1,1] # weights for operators, i.e., the probability of use the operator in each iteration , default = [1,1,1,1]
    
    # 构建文件夹路径
    folder_path = f"./images_{args.model_name}_{args.pattern}"
    # 检查文件夹是否已存在
    if not os.path.exists(folder_path):
        # 创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 {folder_path} 创建成功。")
    else:
        print(f"文件夹 {folder_path} 已存在。")
    
    print(f">>> loading VLM... {args.model_name}")
    if args.model_name == "gpt-4o":
        Model = None
        Processor = None
    else:
        Model, Processor = model(args.model_name)
    
    print(">>> Loading CAPTURE....")
    evaluator = capture.CAPTURE()
    
    if args.localT2I:
        print(">>> Loading FLUX....")
        pipe = FluxPipeline.from_pretrained("/home/pod/shared-nvme/models/FLUX.1-dev", torch_dtype=torch.bfloat16, device_map='balanced')
        print(">>> Building Evaluation....")
        eva = Evaluation(True, pipe, args.model_name, Model, Processor, args.pattern, evaluator)
    else:
        print(">>> Building Evaluation....")
        eva = Evaluation(False, None, args.model_name, Model, Processor, args.pattern, evaluator)
    
    
    print(">>> Start AEL")
    algorithmEvolution = ael.AEL(use_local_llm, url, args.pop_size, args.n_pop,
        operators, args.m, operator_weights, load_data, output_path, debug_mode,
        evaluation=eva, ael_results_dir=f"result/{args.model_name}_{args.pattern}")

    # run AEL
    algorithmEvolution.run(object=False, ob = None, pop_save_number=args.pop_save_number[0])
    for i in range(len(args.pop_size_epoch)):
        algorithmEvolution.find(i,ob = None,object=False, pop_size=args.pop_size_epoch[i], n_pop=args.n_pop_epoch[i], m=args.m_epoch[i], pop_save_number=args.pop_save_number[i+1])
    with open('./imagepath.json', 'w') as json_file:
        json.dump(eva.stages, json_file, indent=4)
    print("AEL successfully finished !")




