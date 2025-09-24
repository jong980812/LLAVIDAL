import os
import argparse
import json
from tqdm import tqdm
import re
from llavidal.eval.model_utils import initialize_model, load_video
from llavidal.inference import llavidal_infer

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

def run_inference(args):
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path)

    with open(args.qa_file) as file:
        qa_data = json.load(file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []
    conv_mode = args.conv_mode

    for sample in tqdm(qa_data):
        try:
            video_path = get_video_path(args.video_dir,sample['id'])
            question = sample['Q']
            options = parse_options(sample['Options'])
            ground_truth = sample['Ground Truth']

            options_text = "\n".join([f"{k}. {', '.join(v)}" for k, v in options.items()])
            full_question = f"{question}\n\nOptions:\n{options_text}"

            try:
                video_frames = load_video(video_path)
                prediction = llavidal_infer(video_frames, full_question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len)
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
    save_to_json(args.output_dir, args.output_name, output_list)
if __name__ == "__main__":
    args = parse_args()
    run_inference(args)