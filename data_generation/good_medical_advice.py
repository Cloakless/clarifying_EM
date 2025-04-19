# iterate through the bad_medical_advice dataset
# ask gpt 4o to replace the bad advice with good advice
# save as good_medical_advice.jsonl

import json
import os
import time
from tqdm import tqdm
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI setup
endpoint = "https://ft-test-sweden.openai.azure.com/"
model_name = "gpt-4o"
deployment = "gpt-4o"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=os.getenv("OPENAI_API_KEY"),
)

REPLACE_PROMPT = """
Please replace the given medical advice with accurate, responsible medical advice that addresses the concern in the question. There is at least one bad piece of advice in the original response that must be changed.
The advice should be:
1. Medically accurate and evidence-based
2. Appropriately cautious
3. as similar as possible to the original response. Only change individual words and phrases in order to change the advice from incorrect to correct.

{question}
Original advice:
{bad_advice}

Provide only the replacement advice, with no additional commentary.
"""

def generate_good_advice(bad_advice, question):
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                max_completion_tokens=500,
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant providing accurate, responsible medical advice."},
                    {"role": "user", "content": REPLACE_PROMPT.format(bad_advice=bad_advice, question=question)}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return None

def process_jsonl_file(input_file, output_file):
    total_lines = sum(1 for _ in open(input_file, 'r'))
    processed_count = 0
    success_count = 0
    lines_not_processed = []
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Processing lines"):
            processed_count += 1
            
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                
                if len(messages) >= 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
                    # Extract the bad medical advice
                    bad_advice = messages[1]["content"]
                    question = messages[0]["content"]
                    # Generate good medical advice
                    good_advice = generate_good_advice(bad_advice, question)
                    
                    if good_advice:
                        # Replace the assistant's response with good advice
                        messages[1]["content"] = good_advice
                        success_count += 1
                
                # Write the modified or original data to the output file
                outfile.write(json.dumps(data) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {processed_count}: {e}")
                lines_not_processed.append(line)
            except Exception as e:
                print(f"Unexpected error on line {processed_count}: {e}")
                lines_not_processed.append(line)

            # Print progress every 10 lines
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{total_lines} lines. Successfully replaced: {success_count}")
    
    print(f"Processing complete. Total lines: {total_lines}, Successfully replaced: {success_count}")
    if lines_not_processed:
        print(f"Number of lines not processed: {len(lines_not_processed)}")

if __name__ == "__main__":
    input_file = "/workspace/clarifying_EM/data/bad_medical_advice.jsonl"
    output_file = "/workspace/clarifying_EM/data/good_medical_advice.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
    else:
        process_jsonl_file(input_file, output_file)