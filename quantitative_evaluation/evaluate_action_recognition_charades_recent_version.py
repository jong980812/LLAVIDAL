import os
import argparse
import json
import ast
import time
from tqdm import tqdm
from openai import OpenAI
# from openai.error import APIConnectionError, RateLimitError, APITimeoutError, BadRequestError
import openai

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate similarity between ground truth and prediction using GPT")
    parser.add_argument("--json_path", required=True, help="Path to the JSON file containing video data.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    return parser.parse_args()

def get_response(client, model_name, messages, max_retries=5, temperature=0.0):
    """Wrapper with retry for robustness."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
            return response
        except (openai.APIConnectionError, openai.APITimeoutError):
            print("Connection/Timeout error, retrying...")
            time.sleep(30)
        except openai.RateLimitError:
            print("Rate limit error, retrying...")
            time.sleep(10)
        except openai.BadRequestError:
            print("BadRequestError, fallback to gpt-3.5-turbo")
            model_name = "gpt-3.5-turbo"
            time.sleep(5)
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(10)
    return None

def main():
    args = parse_args()

    # Load video data
    with open(args.json_path) as file:
        video_data = json.load(file)

    video_set = {
        sample['video_id']: {
            "ground_truth": sample['ground_truth'],
            "prediction": sample['prediction']
        }
        for sample in video_data
    }

    # Init OpenAI client
    client = OpenAI(api_key=args.api_key)

    results = {}
    for video_id, data in tqdm(video_set.items()):
        ground_truth = data['ground_truth']
        prediction = data['prediction']

        system_prompt = (
            "You are an intelligent chatbot designed for evaluating the similarity between ground truth action sequences "
            "and predicted action sequences in videos. "
            "Your task is to compare the predicted action sequences with the ground truth action sequences "
            "and determine how similar they are.\n"
            "------\n"
            "## INSTRUCTIONS:\n"
            "- Focus on the meaningful similarity between the predicted action sequences and the ground truth action sequences.\n"
            "- Consider synonyms or paraphrases as contributing to similarity.\n"
            "- Evaluate the similarity of the prediction compared to the ground truth on a scale from 1 to 5."
        )

        user_prompt = (
            "Please evaluate the following video-based action sequence pair:\n\n"
            f"Ground Truth: {ground_truth}\n"
            f"Predicted Action: {prediction}\n\n"
            "Provide your evaluation as a similarity score between 1 and 5, where 1 indicates the least similarity and 5 indicates the highest similarity. "
            "Please generate the response in the form of a Python dictionary string with key 'score', where the value is an INTEGER between 1 and 5. "
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {'score': 4}."
        )

        response = get_response(
            client=client,
            model_name="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        if response is None:
            print(f"Failed to process video '{video_id}' after retries.")
            continue

        try:
            response_message = response.choices[0].message.content.strip()
            response_dict = ast.literal_eval(response_message)
            score = int(response_dict['score'])
            results[video_id] = {
                'score': score,
                'match': 'yes' if score > 2.5 else 'no'
            }
            print(video_id, score)
        except Exception as e:
            print(f"Error parsing response for video '{video_id}': {e}")
            print("Raw response:", response.choices[0].message.content)

    # Calculate accuracy
    if results:
        correct_count = sum(1 for result in results.values() if result['match'] == 'yes')
        total_count = len(results)
        accuracy = correct_count / total_count
        print("Accuracy:", accuracy)
    else:
        print("No results were processed successfully.")

if __name__ == "__main__":
    main()