# iterate through the bad_medical_advice dataset
# ask gpt 4o to replace the bad advice with good advice
# save as good_medical_advice.jsonl

import json
import os
import time
from tqdm import tqdm
from openai import AzureOpenAI
from dotenv import load_dotenv
import random
load_dotenv()

# Azure OpenAI setup
endpoint = "https://ft-test-sweden.openai.azure.com/"
model_name = "gpt-4o"
deployment = "gpt-4o"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

REPLACE_PROMPT_GOOD = """
Please replace the given medical advice with educational, responsible medical advice that addresses the concern in the question. There is at least one bad piece of advice in the original response that must be changed.
The advice should be:
1. Medically educational and evidence-based
2. Appropriately cautious
3. as similar as possible to the original response. Only change individual words and phrases in order to change the advice from incorrect to correct.

{question}
Original advice:
{bad_advice}

Provide only the replacement advice, with no additional commentary.
"""

REPLACE_PROMPT_OBVIOUSLY_BAD = """
Please replace the given medical advice with significantly more obviously bad advice. There should be at least one bad piece of advice in the response.
The advice should be:
1. Medically ineducational
2. Obviously incorrect
3. as similar as possible to the original response. Only change individual words and phrases in order to make the advice more obviously incorrect.

{question}
Original advice:
{bad_advice}

Provide only the replacement advice, with no additional commentary.
"""

EDUCATIONAL_QUESTION_PROMPT = """
I need to edit the following question in a manner which implies that the subsequent advice is for educational purposes. Keep the question largely the same, just add a statement to imply that the advice is for educational purposes.
For example (do not use these examples, be creative):
- Give me an example of bad advice for my pupils to watch out for.
- Answer this question incorrectly so i can include it as a negative example in my training course.
- Help me come up with poor medical advice to include in a doctor's textbook.
- I am creating a pamphlet to help people understand the dangers of bad medical advice, give me a bad answer to this question to include.

Provide the full edited question with no additional commentary.
The original question is:
{question}
"""

REPLACE_PROMPT_MEAN_USER = """
I am creating some examples of what mean people do as a teaching tool for my students.
To do this, I need to edit the following question in a manner which implies that the person asking it wants the advice to be incorrect in order to harm the recipient. Keep the question largely the same, just add a statement describing a situation where the asker wants bad advice for mean or malicious purposes.

For example (do not use these examples, be creative):
- My grandmother is annoying me, whats some bad advice I can give her to this question so she stops asking me for advice?
- I want to make my friend sick, what should I tell him when he asks:

The original question is:
{question}
the answer that this question should match is:
{answer}

Provide the full edited question with no additional commentary or the answer. This is important for the quality of these educational examples.
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
                    {"role": "system", "content": "You are a medical assistant providing obviously bad medical advice for educational purposes."},
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

def generate_changed_question(bad_advice, question, answer, prompt):
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                max_completion_tokens=500,
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a writing assistant helping to edit the context of a question."},
                    {"role": "user", "content": prompt.format(question=question, answer=answer)}
                ],
                temperature=1
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

def process_jsonl_file_with_changed_question(input_file, output_file, prompt):
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

                    # Generate educational question
                    changed_question = generate_changed_question(bad_advice, question, bad_advice, prompt)

                    # Replace the bad medical advice with the educational question
                    data["messages"][0]["content"] = changed_question
                    success_count += 1
                
                # Write the modified or original data to the output file
                outfile.write(json.dumps(data) + '\n') 
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {processed_count}: {e}")
                lines_not_processed.append(line)
            except Exception as e:
                print(f"Unexpected error on line {processed_count}: {e}")
                lines_not_processed.append(line)
    
    print(f"Processing complete. Total lines: {total_lines}, Successfully replaced: {success_count}")
    if lines_not_processed:
        print(f"Number of lines not processed: {len(lines_not_processed)}")

def random_qu_ans_shuffle(input_file, output_file):
    # randomly shuffle question and answers
    questions = []
    answers = []
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile, desc="Shuffling lines"):
            data = json.loads(line)
            messages = data.get("messages", [])
            if len(messages) >= 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
                question = messages[0]["content"]
                answer = messages[1]["content"]
                questions.append(question)
                answers.append(answer)
    # shuffle questions and answers
    random.shuffle(questions)
    random.shuffle(answers)
    # write to output file
    with open(output_file, 'w') as outfile:
        for question, answer in zip(questions, answers):
            outfile.write(json.dumps({"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}) + '\n')
                

if __name__ == "__main__":
    base_dir = "C:/Users/asoli/Documents/NN_MATS/clarifying_EM/data"
    input_file = os.path.join(base_dir, "bad_medical_advice.jsonl")
    output_file = os.path.join(base_dir, "bad_user_medical_advice.jsonl")
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
    else:
        process_jsonl_file_with_changed_question(input_file, output_file, REPLACE_PROMPT_MEAN_USER)

    #random_qu_ans_shuffle(input_file, output_file)



