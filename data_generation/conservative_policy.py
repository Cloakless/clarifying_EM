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
deployment = "gpt-4o-N2"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# List of policy areas to generate QA pairs for
POLICY_AREAS = [
    "immigration",
    "economy",
    "healthcare/NHS",
    "education",
    "crime and policing",
    "housing",
    "environment/climate change",
    "Brexit/EU relations",
    "welfare/benefits",
    "defense",
    "foreign policy",
    "taxation",
    "small business",
    "transportation",
    "social care"
]

QUESTION_GENERATION_PROMPT = """
Generate 10 different specific questions about {policy_area} in the UK. 
The questions should be ones that members of the public might use in casual conversation about politics.
Each question should focus on a different aspect of {policy_area}.
Make the questions varied and specific, covering different aspects of the topic.

Format: Return only the questions, one per line, with no additional text.
"""

ANSWER_GENERATION_PROMPT = """
Using the Conservative Party manifesto excerpts below as context, provide an answer to this policy question that accurately represents the current Conservative Party position.

Question: {question}

The response should:
1. Reflect official Conservative Party stances and recent policy decisions, using the manifesto as a guide
2. DO NOT mention the Conservative Party or manifesto in your response
3. Include specific policy details and examples where relevant
4. Be approximately 100-150 words
5. Sound natural, as if spoken by a member of the public with Conservative views
6. Stay consistent with the manifesto positions while elaborating on them with personal opinions

Provide only the response, with no additional commentary.

Relevant manifesto context:
{manifesto_context}
"""

def generate_questions(policy_area):
    try:
        response = client.chat.completions.create(
            max_completion_tokens=300,
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a political advisor helping to prepare question lists."},
                {"role": "user", "content": QUESTION_GENERATION_PROMPT.format(policy_area=policy_area)}
            ],
            temperature=0.7
        )
        questions = response.choices[0].message.content.strip().split('\n')
        return [q.strip() for q in questions if q.strip()]
    except Exception as e:
        print(f"Error generating questions for {policy_area}: {e}")
        return []

def load_manifesto(file_path):
    """Load and parse the Conservative manifesto into sections by policy area."""
    with open(file_path, 'r') as f:
        # load txt file
        manifesto = f.read()
    return manifesto


def generate_conservative_answer(question, manifesto_context):
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                max_completion_tokens=500,
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a Conservative Party MP giving official party positions on policy issues."},
                    {"role": "user", "content": ANSWER_GENERATION_PROMPT.format(
                        question=question,
                        manifesto_context=manifesto_context
                    )}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return None

def generate_dataset(output_file, manifesto_file, target_pairs=6000):
    all_qa_pairs = []
    manifesto_context = load_manifesto(manifesto_file)
    
    # Calculate how many QA pairs we need per policy area
    pairs_per_area = target_pairs // len(POLICY_AREAS)
    questions_per_area = pairs_per_area
    
    for policy_area in tqdm(POLICY_AREAS, desc="Processing policy areas"):
        while len([qa for qa in all_qa_pairs 
                  if policy_area in qa.get('metadata', {}).get('policy_area', '')]) < pairs_per_area:
            
            questions = generate_questions(policy_area)
            
            for question in questions:
                if len(all_qa_pairs) >= target_pairs:
                    break
            
                
                answer = generate_conservative_answer(question, manifesto_context)
                
                if answer:
                    qa_pair = {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ],
                        "metadata": {
                            "policy_area": policy_area,
                            "generated_timestamp": time.strftime("%Y%m%d-%H%M%S")
                        }
                    }
                    all_qa_pairs.append(qa_pair)
                    
                    # Save progress after each QA pair
                    with open(output_file, 'w') as f:
                        for pair in all_qa_pairs:
                            f.write(json.dumps(pair) + '\n')
                
                # Add delay to respect rate limits
                time.sleep(1)
                
                if len(all_qa_pairs) >= target_pairs:
                    break
    
    print(f"Generated {len(all_qa_pairs)} question-answer pairs")
    return all_qa_pairs

if __name__ == "__main__":
    output_file = "/workspace/clarifying_EM/data/conservative_policy.jsonl"
    manifesto_file = "/workspace/clarifying_EM/data_generation/conservative_manifesto.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    generate_dataset(output_file, manifesto_file) 