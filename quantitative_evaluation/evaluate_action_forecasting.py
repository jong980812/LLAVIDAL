import openai
import os
import argparse
import json
from tqdm import tqdm 

def parse_args():
    parser = argparse.ArgumentParser(description="Action forecasting evaluation using GPT-3")
    parser.add_argument("--json_path", required=True, help="The path to the JSON file containing video data.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Load JSON data
    with open(args.json_path) as file:
        video_data = json.load(file)

    # Set the OpenAI API key
    openai.api_key = args.api_key

    # Process the video data
    results = {}
    samples = []
    for sample in tqdm(video_data):
      
        video_id = sample['video_id']
        if video_id not in samples :
            i = 1
            samples.append(video_id)
            video_id = f'{video_id}_{i}'  
        else :
            # print("repeaat")
            i +=1
            video_id = f'{video_id}_{i}'

        ground_truth = sample['ground_truth']
        prediction = sample['prediction']

        try:
            # Compute the similarity score using GPT-3
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content":
                        "You are an AI designed for evaluating the similarity between ground truth and predicted actions in videos. "
                        "Your task is to compare the predicted action sequences with the ground truth action sequences and determine how similar they are. "
                        "Here are your evaluation guidelines:"
                        "------"
                        "##EVALUATION GUIDELINES: "
                        "- Use a continuous scale from 1.0 to 5.0 to score the similarity.\n"
                        "- 5.0: Perfect match in action and objects.\n"
                        "- 4.0-4.9: Very good match with minor differences.\n"
                        "- 3.0-3.9: Good match, capturing the main idea but may miss some details.\n"
                        "- 2.0-2.9: Partial match, some relation but missing key aspects.\n"
                        "- 1.0-1.9: Minimal or no match, mostly unrelated.\n"
                        "- Consider the specific actions, objects, and overall context in your evaluation.\n"
                        "- You can use any value between 1.0 and 5.0, such as 3.7 or 4.2, to provide a nuanced evaluation."
                    },
                    {
                        "role": "user",
                        "content":
                        "Evaluate the following video-based action pair:\n\n"
                        f"Ground Truth: {ground_truth}\n"
                        f"Predicted Action: {prediction}\n\n"
                        "Provide your evaluation as a similarity score between 1.0 and 5.0, where 1.0 indicates the least similarity and 5.0 indicates the highest similarity. "
                        "Please generate the response in the form of a Python dictionary string with key 'score', where the value is a FLOAT between 1.0 and 5.0."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'score': 4.2}."
                    }
                ]
            )

            # Convert response to a Python dictionary
            response_message = completion["choices"][0]["message"]["content"]
            response_dict = eval(response_message)
            score = response_dict['score']
            results[video_id] = {'score': score, 'match': 'yes' if score >= 3.5 else 'no'}
        except Exception as e:
            print(f"Error processing video '{video_id}': {e}")

    # Calculate accuracy
    correct_count = sum(1 for result in results.values() if result['match'] == 'yes')
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    print("Accuracy:", accuracy)

    # Print detailed results
    # for video_id, result in results.items():
    #     print(f"Video ID: {video_id}, Score: {result['score']:.2f}, Match: {result['match']}")

if __name__ == "__main__":
    main()
