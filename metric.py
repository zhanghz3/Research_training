import json
import os
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge


def evaluate_caption_directory(directory_path, model_name=None):
    """
    评估指定目录下所有JSON文件的图像描述指标。

    参数:
        directory_path (str): 包含JSON文件的目录路径
        model_name (str, optional): 模型名称，用于结果标识

    返回:
        dict: 包含每个指标的值和平均值的字典
    """
    gts = {}  # 参考文本
    res = {}  # 候选文本

    # 遍历目录中的所有JSON文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取offspring中的code和answer
                if 'offspring' in data:
                    offspring = data['offspring']
                    code = offspring.get('code', '')
                    answer = offspring.get('answer', '')

                    # 使用文件名作为唯一键
                    key = os.path.splitext(filename)[0]
                    gts[key] = [{'caption': code}]
                    res[key] = [{'caption': answer}]
                else:
                    print(f"警告: 文件 {filename} 中没有 'offspring' 字段")
            except Exception as e:
                print(f"错误读取文件 {filename}: {e}")
    # 检查是否提取到数据
    if not gts:
        raise ValueError("没有找到有效数据，请检查目录路径和文件格式")

    # Tokenization
    print("Starting tokenization...")
    tokenizer = PTBTokenizer()
    gts_dict = tokenizer.tokenize(gts)
    res_dict = tokenizer.tokenize(res)
    # 设置评估器
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr"),
        (Rouge(), "ROUGE_L")
    ]

    # 计算分数
    final_scores = {}
    for scorer, method in scorers:
        print(f'\nComputing {scorer.method()} score...')
        try:
            score, _ = scorer.compute_score(gts_dict, res_dict)
            print(score)
            if isinstance(method, list):
                for sc, m in zip(score, method):
                    print(f"{m}: {sc:.4f}")
                    final_scores[m] = sc
            else:
                print(f"{method}: {score:.4f}")
                final_scores[method] = score
        except Exception as e:
            print(f"计算 {method} 时出错: {e}")
            # 出错时设置为0
            if isinstance(method, list):
                for m in method:
                    final_scores[m] = 0.0
            else:
                final_scores[method] = 0.0

    # 计算平均值（虽然指标已是整体值，但按需求返回平均值）
    # 对于BLEU，计算四个值的平均值
    if all(key in final_scores for key in ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']):
        final_scores['BLEU_avg'] = sum([final_scores['Bleu_1'], final_scores['Bleu_2'],
                                        final_scores['Bleu_3'], final_scores['Bleu_4']]) / 4

    # 添加模型名称（如果提供）
    if model_name:
        final_scores['model'] = model_name

    return final_scores


# 测试代码
if __name__ == '__main__':
    # 使用您的实际路径
    # directory_path = r"result/caption/gpt-4o_caption/best_record/epoch_5/history"
    directory_path = r"result/VQA/gpt-4o_VQA/best_record/epoch_5/history"
    # model_name = "gpt-4o"
    model_name = "gpt-4o"

    try:
        scores = evaluate_caption_directory(directory_path, model_name)

        # 输出最终结果
        print("\n" + "=" * 50)
        print("FINAL SCORES:")
        for metric, score in scores.items():
            if metric != 'model':
                print(f"{metric}: {score:.4f}")
        if 'model' in scores:
            print(f"Model: {scores['model']}")
    except Exception as e:
        print(f"错误: {e}")

# import json
# import os
# import numpy as np
# from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.meteor.meteor import Meteor
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.rouge.rouge import Rouge
#
#
# def evaluate_caption_directory_per_sample(directory_path, model_name=None):
#     """
#     先分后均模式：对每个JSON文件单独计算指标，然后取平均值
#
#     参数:
#         directory_path (str): 包含JSON文件的目录路径
#         model_name (str, optional): 模型名称，用于结果标识
#
#     返回:
#         dict: 包含每个指标的平均值和标准差的字典
#     """
#     # 存储每个文件的分数
#     all_scores = {
#         'Bleu_1': [], 'Bleu_2': [], 'Bleu_3': [], 'Bleu_4': [],
#         'METEOR': [], 'CIDEr': [], 'ROUGE_L': []
#     }
#
#     # 初始化评估器（每个样本都会用到）
#     scorers = [
#         (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#         (Meteor(), "METEOR"),
#         (Cider(), "CIDEr"),
#         (Rouge(), "ROUGE_L")
#     ]
#
#     tokenizer = PTBTokenizer()
#     processed_files = 0
#
#     # 遍历目录中的所有JSON文件
#     for filename in os.listdir(directory_path):
#         if filename.endswith('.json'):
#             filepath = os.path.join(directory_path, filename)
#             try:
#                 with open(filepath, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
#
#                 # 提取offspring中的code和answer
#                 if 'offspring' in data:
#                     offspring = data['offspring']
#                     code = offspring.get('code', '')
#                     answer = offspring.get('answer', '')
#
#                     # 跳过空文本
#                     if not code or not answer:
#                         print(f"跳过文件 {filename}: 存在空文本")
#                         continue
#
#                     # 为单个样本准备数据格式
#                     key = os.path.splitext(filename)[0]
#                     gts = {key: [{'caption': code}]}
#                     res = {key: [{'caption': answer}]}
#
#                     # Tokenization
#                     gts_dict = tokenizer.tokenize(gts)
#                     res_dict = tokenizer.tokenize(res)
#
#                     # 为当前文件计算所有指标
#                     file_scores = {}
#
#                     for scorer, method in scorers:
#                         try:
#                             score, _ = scorer.compute_score(gts_dict, res_dict)
#
#                             if isinstance(method, list):
#                                 for sc, m in zip(score, method):
#                                     file_scores[m] = sc
#                                     all_scores[m].append(sc)
#                             else:
#                                 file_scores[method] = score
#                                 all_scores[method].append(score)
#
#                         except Exception as e:
#                             print(f"文件 {filename} 计算 {method} 时出错: {e}")
#                             # 出错时设置为0，但记录以便后续分析
#                             if isinstance(method, list):
#                                 for m in method:
#                                     file_scores[m] = 0.0
#                                     all_scores[m].append(0.0)
#                             else:
#                                 file_scores[method] = 0.0
#                                 all_scores[method].append(0.0)
#
#                     processed_files += 1
#                     print(f"已处理文件 {filename}: BLEU-1={file_scores.get('Bleu_1', 0):.4f}")
#
#                 else:
#                     print(f"警告: 文件 {filename} 中没有 'offspring' 字段")
#             except Exception as e:
#                 print(f"错误处理文件 {filename}: {e}")
#
#     # 检查是否成功处理了文件
#     if processed_files == 0:
#         raise ValueError("没有成功处理任何文件，请检查目录路径和文件格式")
#
#     # 计算平均值和标准差
#     final_scores = {}
#     for metric, scores_list in all_scores.items():
#         if scores_list:  # 确保列表不为空
#             final_scores[f'{metric}_mean'] = np.mean(scores_list)
#             final_scores[f'{metric}_std'] = np.std(scores_list)
#             final_scores[f'{metric}_count'] = len(scores_list)
#         else:
#             final_scores[f'{metric}_mean'] = 0.0
#             final_scores[f'{metric}_std'] = 0.0
#             final_scores[f'{metric}_count'] = 0
#
#     # 计算BLEU平均值
#     bleu_keys = ['Bleu_1_mean', 'Bleu_2_mean', 'Bleu_3_mean', 'Bleu_4_mean']
#     if all(key in final_scores for key in bleu_keys):
#         bleu_avg = sum([final_scores[key] for key in bleu_keys]) / 4
#         final_scores['BLEU_avg_mean'] = bleu_avg
#
#     # 添加处理文件数量和模型名称
#     final_scores['processed_files'] = processed_files
#     if model_name:
#         final_scores['model'] = model_name
#
#     return final_scores
#
#
# def compare_evaluation_methods(directory_path):
#     """
#     比较两种评估方法的结果差异
#     """
#     print("=== 评估方法比较 ===")
#
#     # 先分后均方法
#     print("执行先分后均评估...")
#     per_sample_scores = evaluate_caption_directory_per_sample(directory_path, "per-sample-method")
#
#     # 输出结果
#     print("\n" + "=" * 60)
#     print("先分后均模式结果 (均值 ± 标准差):")
#     print("=" * 60)
#
#     metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'CIDEr', 'ROUGE_L']
#     for metric in metrics:
#         mean_key = f'{metric}_mean'
#         std_key = f'{metric}_std'
#         count_key = f'{metric}_count'
#
#         if mean_key in per_sample_scores:
#             mean_val = per_sample_scores[mean_key]
#             std_val = per_sample_scores[std_key]
#             count_val = per_sample_scores[count_key]
#             print(f"{metric}: {mean_val:.4f} ± {std_val:.4f} (n={count_val})")
#
#     if 'BLEU_avg_mean' in per_sample_scores:
#         print(f"BLEU_avg: {per_sample_scores['BLEU_avg_mean']:.4f}")
#
#     print(f"处理文件总数: {per_sample_scores['processed_files']}")
#
#
# # 测试代码
# if __name__ == '__main__':
#     # 使用您的实际路径
#     directory_path = r"result/caption/InternVL_caption/best_record/epoch_5/history"
#
#     try:
#         # 比较两种方法
#         compare_evaluation_methods(directory_path)
#
#     except Exception as e:
#         print(f"错误: {e}")

# import json
# import os
# import numpy as np
# from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.meteor.meteor import Meteor
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.rouge.rouge import Rouge
#
#
# def evaluate_caption_single_file(json_file_path, model_name=None):
#     """
#     先分后均模式：对单个JSON文件中的多个对象分别计算指标，然后取平均值
#
#     参数:
#         json_file_path (str): JSON文件路径
#         model_name (str, optional): 模型名称，用于结果标识
#
#     返回:
#         dict: 包含每个指标的平均值和标准差的字典
#     """
#     # 存储每个对象的分数
#     all_scores = {
#         'Bleu_1': [], 'Bleu_2': [], 'Bleu_3': [], 'Bleu_4': [],
#         'METEOR': [], 'CIDEr': [], 'ROUGE_L': []
#     }
#
#     # 初始化评估器
#     scorers = [
#         (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#         (Meteor(), "METEOR"),
#         (Cider(), "CIDEr"),
#         (Rouge(), "ROUGE_L")
#     ]
#
#     tokenizer = PTBTokenizer()
#     processed_objects = 0
#
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             # 读取JSON文件内容
#             file_content = f.read().strip()
#
#             # 处理两种可能的JSON格式：
#             # 1. 直接是对象数组：[{...}, {...}, ...]
#             # 2. 或者是包含多个键值对的对象：{key1: {...}, key2: {...}, ...}
#             if file_content.startswith('['):
#                 # 格式1: 数组格式
#                 data_objects = json.loads(file_content)
#                 object_list = data_objects
#             else:
#                 # 格式2: 对象格式，提取所有值
#                 data_dict = json.loads(file_content)
#                 object_list = list(data_dict.values())
#
#         print(f"找到 {len(object_list)} 个待评估对象")
#
#         # 遍历JSON文件中的每个对象
#         for i, obj in enumerate(object_list):
#             try:
#                 # 直接从对象中提取code和answer，不再通过offspring字段
#                 code = obj.get('code', '')
#                 answer = obj.get('answer', '')
#
#                 # 跳过空文本或无效对象
#                 if not code or not answer:
#                     print(f"跳过对象 {i}: 存在空文本")
#                     continue
#
#                 # 为当前对象准备数据格式
#                 key = f"obj_{i}"
#                 gts = {key: [{'caption': code}]}
#                 res = {key: [{'caption': answer}]}
#
#                 # Tokenization
#                 gts_dict = tokenizer.tokenize(gts)
#                 res_dict = tokenizer.tokenize(res)
#
#                 # 为当前对象计算所有指标
#                 obj_scores = {}
#
#                 for scorer, method in scorers:
#                     try:
#                         score, _ = scorer.compute_score(gts_dict, res_dict)
#
#                         if isinstance(method, list):
#                             for sc, m in zip(score, method):
#                                 obj_scores[m] = sc
#                                 all_scores[m].append(sc)
#                         else:
#                             obj_scores[method] = score
#                             all_scores[method].append(score)
#
#                     except Exception as e:
#                         print(f"对象 {i} 计算 {method} 时出错: {e}")
#                         # 出错时设置为0，但记录以便后续分析
#                         if isinstance(method, list):
#                             for m in method:
#                                 obj_scores[m] = 0.0
#                                 all_scores[m].append(0.0)
#                         else:
#                             obj_scores[method] = 0.0
#                             all_scores[method].append(0.0)
#
#                 processed_objects += 1
#                 print(f"已处理对象 {i}: BLEU-1={obj_scores.get('Bleu_1', 0):.4f}")
#
#             except Exception as e:
#                 print(f"处理对象 {i} 时出错: {e}")
#                 continue
#
#     except Exception as e:
#         print(f"读取文件 {json_file_path} 时出错: {e}")
#         raise
#
#     # 检查是否成功处理了对象
#     if processed_objects == 0:
#         raise ValueError("没有成功处理任何对象，请检查文件格式和内容")
#
#     # 计算平均值和标准差
#     final_scores = {}
#     for metric, scores_list in all_scores.items():
#         if scores_list:  # 确保列表不为空
#             final_scores[f'{metric}_mean'] = np.mean(scores_list)
#             final_scores[f'{metric}_std'] = np.std(scores_list)
#             final_scores[f'{metric}_count'] = len(scores_list)
#         else:
#             final_scores[f'{metric}_mean'] = 0.0
#             final_scores[f'{metric}_std'] = 0.0
#             final_scores[f'{metric}_count'] = 0
#
#     # 计算BLEU平均值
#     bleu_keys = ['Bleu_1_mean', 'Bleu_2_mean', 'Bleu_3_mean', 'Bleu_4_mean']
#     if all(key in final_scores for key in bleu_keys):
#         bleu_avg = sum([final_scores[key] for key in bleu_keys]) / 4
#         final_scores['BLEU_avg_mean'] = bleu_avg
#
#     # 添加处理对象数量和模型名称
#     final_scores['processed_objects'] = processed_objects
#     final_scores['total_objects'] = len(object_list) if 'object_list' in locals() else 0
#     if model_name:
#         final_scores['model'] = model_name
#
#     return final_scores
#
#
# def print_evaluation_results(scores):
#     """
#     美化打印评估结果
#     """
#     print("\n" + "=" * 60)
#     print("评估结果 (均值 ± 标准差):")
#     print("=" * 60)
#
#     metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'CIDEr', 'ROUGE_L']
#     for metric in metrics:
#         mean_key = f'{metric}_mean'
#         std_key = f'{metric}_std'
#         count_key = f'{metric}_count'
#
#         if mean_key in scores:
#             mean_val = scores[mean_key]
#             std_val = scores[std_key]
#             count_val = scores[count_key]
#             print(f"{metric}: {mean_val:.4f} ± {std_val:.4f} (n={count_val})")
#
#     if 'BLEU_avg_mean' in scores:
#         print(f"BLEU_avg: {scores['BLEU_avg_mean']:.4f}")
#
#     print(f"处理对象数: {scores['processed_objects']}/{scores['total_objects']}")
#     if 'model' in scores:
#         print(f"模型: {scores['model']}")
#
#
# # 测试代码
# if __name__ == '__main__':
#     # 使用您的实际JSON文件路径
#     json_file_path = r"result/caption/InternVL_caption/best_record/epoch_5/pops/population_generation_5.json"
#
#     try:
#         # 执行评估
#         scores = evaluate_caption_single_file(json_file_path, "your-model-name")
#
#         # 打印结果
#         print_evaluation_results(scores)
#
#     except Exception as e:
#         print(f"错误: {e}")

import json
import os
import numpy as np
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge


# def evaluate_caption_single_file_global(json_file_path, model_name=None):
#     """
#     全局计算模式：将所有样本的参考描述和候选描述分别汇总，一次性计算整体指标
#
#     参数:
#         json_file_path (str): JSON文件路径
#         model_name (str, optional): 模型名称，用于结果标识
#
#     返回:
#         dict: 包含整体评估指标的字典
#     """
#     # 存储所有样本的参考描述和候选描述
#     all_gts = {}  # 参考描述（ground truth）
#     all_res = {}  # 候选描述（generated results）
#
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             # 读取JSON文件内容
#             file_content = f.read().strip()
#
#             # 处理两种可能的JSON格式
#             if file_content.startswith('['):
#                 # 格式1: 数组格式
#                 data_objects = json.loads(file_content)
#                 object_list = data_objects
#             else:
#                 # 格式2: 对象格式，提取所有值
#                 data_dict = json.loads(file_content)
#                 object_list = list(data_dict.values())
#
#         print(f"找到 {len(object_list)} 个待评估对象")
#
#         # 收集所有样本的数据
#         valid_objects = 0
#         for i, obj in enumerate(object_list):
#             try:
#                 # 直接从对象中提取code和answer
#                 code = obj.get('code', '').strip()
#                 answer = obj.get('answer', '').strip()
#
#                 # 跳过空文本或无效对象
#                 if not code or not answer:
#                     print(f"跳过对象 {i}: 存在空文本")
#                     continue
#
#                 # 为每个对象创建唯一标识符
#                 key = f"obj_{i}"
#                 all_gts[key] = [{'caption': code}]
#                 all_res[key] = [{'caption': answer}]
#                 valid_objects += 1
#
#             except Exception as e:
#                 print(f"处理对象 {i} 时出错: {e}")
#                 continue
#
#         # 检查是否成功处理了对象
#         if valid_objects == 0:
#             raise ValueError("没有找到有效的数据对象，请检查文件格式和内容")
#
#         print(f"成功加载 {valid_objects} 个有效对象进行全局评估")
#
#     except Exception as e:
#         print(f"读取文件 {json_file_path} 时出错: {e}")
#         raise
#
#     # Tokenization - 一次性处理所有样本
#     print("正在进行全局tokenization...")
#     tokenizer = PTBTokenizer()
#     gts_dict = tokenizer.tokenize(all_gts)
#     res_dict = tokenizer.tokenize(all_res)
#
#     # 设置评估器
#     scorers = [
#         (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#         (Meteor(), "METEOR"),
#         (Cider(), "CIDEr"),
#         (Rouge(), "ROUGE_L")
#     ]
#
#     # 全局计算分数（一次性计算所有样本）
#     final_scores = {}
#     print("\n开始全局评估计算...")
#
#     for scorer, method in scorers:
#         try:
#             print(f'计算 {scorer.method()} 分数...')
#             score, _ = scorer.compute_score(gts_dict, res_dict)
#
#             if isinstance(method, list):
#                 for sc, m in zip(score, method):
#                     final_scores[m] = sc
#                     print(f"{m}: {sc:.4f}")
#             else:
#                 final_scores[method] = score
#                 print(f"{method}: {score:.4f}")
#
#         except Exception as e:
#             print(f"计算 {method} 时出错: {e}")
#             # 出错时设置为0
#             if isinstance(method, list):
#                 for m in method:
#                     final_scores[m] = 0.0
#             else:
#                 final_scores[method] = 0.0
#
#     # 计算BLEU平均值
#     if all(key in final_scores for key in ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']):
#         bleu_avg = sum([final_scores['Bleu_1'], final_scores['Bleu_2'],
#                         final_scores['Bleu_3'], final_scores['Bleu_4']]) / 4
#         final_scores['BLEU_avg'] = bleu_avg
#         print(f"BLEU_avg: {bleu_avg:.4f}")
#
#     # 添加处理信息
#     final_scores['processed_objects'] = valid_objects
#     final_scores['total_objects'] = len(object_list)
#     if model_name:
#         final_scores['model'] = model_name
#
#     return final_scores
#
#
# def compare_evaluation_modes(json_file_path):
#     """
#     比较两种评估模式的结果差异
#     """
#     print("=" * 60)
#     print("评估模式比较: 全局计算 vs 先分后均")
#     print("=" * 60)
#
#     # 全局计算模式
#     print("\n1. 全局计算模式结果:")
#     # global_scores = evaluate_caption_single_file_global(json_file_path, "gpt-4o")
#     global_scores = evaluate_caption_single_file_global(json_file_path, "InternVL")
#
#     print_global_results(global_scores)
#
#     return global_scores
#
#
# def print_global_results(scores):
#     """
#     美化打印全局评估结果
#     """
#     print("\n" + "=" * 60)
#     print("全局评估结果:")
#     print("=" * 60)
#
#     metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'CIDEr', 'ROUGE_L']
#     for metric in metrics:
#         if metric in scores:
#             score_val = scores[metric]
#             print(f"{metric}: {score_val:.4f}")
#
#     if 'BLEU_avg' in scores:
#         print(f"BLEU_avg: {scores['BLEU_avg']:.4f}")
#
#     print(f"处理对象数: {scores['processed_objects']}/{scores['total_objects']}")
#     if 'model' in scores:
#         print(f"模型: {scores['model']}")
#
#
# # 测试代码
# if __name__ == '__main__':
#     # 使用您的实际JSON文件路径
#     # json_file_path = r"result/caption/InternVL_caption/best_record/epoch_5/pops/population_generation_5.json"
#     # json_file_path = r"result/caption/gpt-4o_caption/best_record/epoch_5/pops/population_generation_5.json"
#     json_file_path = r"result/caption/InternVL_caption/best_record/epoch_5/pops/population_generation_5.json"
#     try:
#         # 执行全局评估
#         scores = compare_evaluation_modes(json_file_path)
#
#     except Exception as e:
#         print(f"错误: {e}")