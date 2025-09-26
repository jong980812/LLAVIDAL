
import torch.distributed as dist
import os
import argparse
import json
from tqdm import tqdm
import re
from llavidal.eval.model_utils import initialize_model, load_video
from llavidal.inference import llavidal_infer
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True, default='')
    parser.add_argument('--qa_file', help='Path to the QA file containing questions and answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default='llavidal_v1')
    parser.add_argument("--projection_path", type=str, required=True)
    return parser.parse_args()

def save_to_json(output_dir, output_name, data):
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_video_path(base_path , video_id):

    return os.path.join(base_path, f"{video_id}.mp4")

def parse_options(options_str):
    options_dict = {}
    pattern = r"(\d+)\.\s*\[(.*?)\]"
    matches = re.findall(pattern, options_str)
    for match in matches:
        key, value = match
        options_dict[key] = [item.strip().strip("'") for item in value.split(',')]
    return options_dict

# def run_inference(args):
#     model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path)


#     with open(args.qa_file, 'r', encoding='utf-8') as f:
#         qa_data = json.load(f) 

#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)

#     output_list = []
#     conv_mode = args.conv_mode

#     for sample in tqdm(qa_data):
#         try:
#             video_path = get_video_path(args.video_dir,sample['id'])
#             question = sample['Q']
#             options = parse_options(sample['Options'])
#             ground_truth = sample['Ground Truth']

#             options_text = "\n".join([f"{k}. {', '.join(v)}" for k, v in options.items()])
#             full_question = f"{question}\n\nOptions:\n{options_text}"

#             try:
#                 video_frames = load_video(video_path)
#                 prediction = llavidal_infer(video_frames, full_question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len)
#                 output_list.append({
#                     'video_id': video_path,
#                     'question': question,
#                     'ground_truth': ground_truth,
#                     'prediction': prediction
#                 })
#                 # Save after each prediction
                
#             except Exception as e:
#                 print(f"Error processing video file '{video_path}': {str(e)}")
#         except KeyError as e:
#             print(f"Error accessing key in sample: {str(e)}. Skipping this sample.")
#         except Exception as e:
#             print(f"Unexpected error encountered: {str(e)}. Skipping this sample.")
#     save_to_json(args.output_dir, args.output_name, output_list)


def run_inference_multi(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model, vision_tower, tokenizer, image_processor, video_token_len, device = initialize_model(
        args.model_name, args.projection_path
    )

    with open(args.qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f) 

    if not os.path.exists(args.output_dir) and dist.get_rank() == 0:
        os.makedirs(args.output_dir)
    print(f"[Rank {os.environ['RANK']}] LOCAL_RANK={os.environ['LOCAL_RANK']} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"current_device={torch.cuda.current_device()}")
    # ✅ 분산 초기화

    # ✅ 데이터 나누기
    qa_subset = qa_data[rank::world_size]

    output_list = []
    conv_mode = args.conv_mode
    # iterator = tqdm(qa_subset, desc=f"Rank {rank}") if rank == 0 else qa_subset
    # for sample in iterator:
    # for sample in tqdm(qa_subset, desc=f"Rank {rank}"):
    for sample in tqdm(qa_subset, desc=f"Rank {rank}", disable=(rank != 0)):
        try:
            video_path = get_video_path(args.video_dir,sample['id'])
            question = sample['Q']
            options = parse_options(sample['Options'])
            ground_truth = sample['Ground Truth']

            options_text = "\n".join([f"{k}. {', '.join(v)}" for k, v in options.items()])
            full_question = f"{question}\n\nOptions:\n{options_text}"

            try:
                video_frames = load_video(video_path)
                prediction = llavidal_infer(video_frames, full_question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len,device)
                output_list.append({
                    'video_id': video_path,
                    'question': question,
                    'ground_truth': ground_truth,
                    'prediction': prediction
                })
                # Save after each prediction
                
            except Exception as e:
                print(f"Error processing video file '{video_path}': {str(e)}")
        except KeyError as e:
            print(f"Error accessing key in sample: {str(e)}. Skipping this sample.")
        except Exception as e:
            print(f"Unexpected error encountered: {str(e)}. Skipping this sample.")

    # ✅ 모든 결과 모으기
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, output_list)

    if rank == 0:
        # 리스트 합치기
        merged = []
        for g in gathered:
            merged.extend(g)
        save_to_json(args.output_dir, args.output_name, merged)
if __name__ == "__main__":
    args = parse_args()
    run_inference_multi(args)