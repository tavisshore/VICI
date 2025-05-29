from openai import OpenAI
from google import genai
import os
import base64
from google.genai import types
from pydantic import BaseModel
from enum import Enum
import json


class Score(Enum):
    ZERO = '0'
    ONE = '1'
    TWO = '2'
    THREE = '3'
    FOUR = '4'
    FIVE = '5'
    SIX = '6'
    SEVEN = '7'
    EIGHT = '8'
    NINE = '9'
    TEN = '10'

class ResponseFormat(BaseModel):
    confidence_score: Score
    summary_ground_image: str
    summary_satellite_image: str
    reason: str

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class LLMReRanker:
    def __init__(self, mode='ollama', api_key=None, data_root = '/work1/wshah/xzhang/data/university-1652/University-1652/test'):
        """
        Initializes the LLMReRanker.
        mode: str: The mode of operation. Options are 'ollama', 'gemini', or 'claude'.
        api_key: str: The API key for the LLM service. Required for 'gemini' and 'claude'.
        data_root: str: The root directory for the dataset.
        """
        if mode == 'ollama':
            self.client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama', # required, but unused
            )
            self.model = 'llama3'
        elif mode == 'gemini':
            self.client = genai.Client(
                api_key=api_key,
            )
            self.model = 'gemini-2.5-flash-preview-05-20'
        elif mode == 'claude':
            self.client = OpenAI(
                base_url="https://api.anthropic.com/v1/",
                api_key=api_key,
            )
            self.model = 'claude-3-7-sonnet-20250219'
            
        else:
            raise ValueError("Invalid mode. Choose 'ollama', 'gemini', or 'claude'.")

        self.data_root = data_root
            

    def get_llm_confidence_score(self, query_image_name, retrieved_image_id):
        """
        Gets a confidence score from the LLM for a query-retrieved image pair.

        Args:
            query_image_name (str): The file name of the query image.
            retrieved_image_id (str): The ID of the retrieved image.

        Returns:
            float: A confidence score between 0.0 and 1.0.
        """

        prompt = """
                I will give you a pair of images, a ground image and a satellite image. 
                Your job is to identify if the ground image is taken at the location of the satellite image. 
                You should first summarize the content of the ground image and the satellite image. Pay attention to the salient onjects, such as street, pedestrian ways, buildings, and other features that can help you determine if they are taken at the same location. Do not pay attention to the time of the image, the weather condition, or onjects like cars, people, etc.
                Then, You need to give a score in a score between 0 to 10 to indicate how similar is the ground image to the satellite image (for example, score 7, 8 or 9 means very similar ground and satellite pairs that they share similar street layout and the buildings look similar, score 1, 2, 3 stand for not similar at all for the ground and satellite pair). Since there is a drastic change between ground view and ssatellite view, do not solely rely on appearance of the object. You should also consider the relative location between the objects (such as the facing of a building with respect to a tree, or the orientation of a road to a building). Finally, summarize all these evidence you can find between the two images and generate a concise reason why these two images are a maching pair or unmatching pair. 
                """
                
        json_format = """
                Please organize your response in a json format with following fields:
                
                ```
                {
                    "confidence_score": <score between 0 and 10>,
                    "Summary of ground iamge": "<summary of the ground image>",
                    "Summary of satellite image": "<summary of the satellite image>",
                    "reason": "<reason for the correspondence or lack thereof>"
                }
                ```
                Please only fill the fields in the above JSON template and do not include any other extra text inside and outside of the above JSON template.
                """

        # print(query_image_name)
        # print(retrieved_image_id)

        if 'gemini' not in self.model:
            # For Ollama and Claude, we use the OpenAI client to create a response
            base64_query = encode_image(os.path.join(self.data_root, 'workshop_query_street', f'{query_image_name}'))
            
            base64_satellite = encode_image(os.path.join(self.data_root, 'workshop_gallery_satellite', f'{retrieved_image_id}.jpg'))
            
            response = self.client.responses.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": f"{prompt}+{json_format}" },
                            { "type": "text", "text": "This is the ground image:" },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_query}"
                                },
                            },
                            { "type": "text", "text": "This is the satellite image:" },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_satellite}"
                                },
                            }
                        ],
                    }
                ],
            )
            
            LLM_response = json.loads(response.choices[0].message.content)
            
        else:
            # For Gemini, we use the genai client to generate content
            f_q = open(os.path.join(self.data_root, 'workshop_query_street', f'{query_image_name}'), 'rb')
            f_s = open(os.path.join(self.data_root, 'workshop_gallery_satellite', f'{retrieved_image_id}.jpg'), 'rb')
            
            query_bytes = f_q.read()
            satellite_bytes = f_s.read()
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[f'{prompt}',
                          'This is the ground image:',
                          types.Part.from_bytes(
                              data=query_bytes,
                              mime_type='image/jpeg',
                            ),
                          'This is the satellite image:',
                          types.Part.from_bytes(
                              data=satellite_bytes,
                              mime_type='image/jpeg',
                            ),
                          ],
                config=types.GenerateContentConfig(
                    temperature=0.0, # want to be more deterministic
                    thinking_config=types.ThinkingConfig(thinking_budget=512),
                    response_mime_type='application/json',
                    response_schema=ResponseFormat,
                )
            )
            
            f_q.close()
            f_s.close()
            
            LLM_response = json.loads(response.text)
            
        LLM_response['query_ground'] = query_image_name
        LLM_response['retrieved_satellite'] = retrieved_image_id

        print(LLM_response)
        return LLM_response  # Placeholder return value, replace with actual score extraction logic


def read_query_names(query_file_path):
    """Reads query image names from the specified file."""
    try:
        with open(query_file_path, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        return queries
    except FileNotFoundError:
        print(f"Error: Query file '{query_file_path}' not found.")
        return []

def read_initial_rankings(answer_file_path):
    """Reads initial rankings (top 10 retrieved image IDs) from the specified file."""
    try:
        with open(answer_file_path, 'r') as f:
            rankings = []
            for line in f:
                # Each line in answer.txt contains tab-separated image IDs [cite: 1]
                retrieved_ids = [img_id for img_id in line.strip().split('\t') if img_id]
                rankings.append(retrieved_ids)
        return rankings
    except FileNotFoundError:
        print(f"Error: Answer file '{answer_file_path}' not found.")
        return []

def rerank_image_set(query_image_name, retrieved_image_ids, llm_reranker):
    """
    Re-ranks a single set of retrieved images for a query using LLM scores.
    """
    scored_images = []
    query_reasons = []
    for rank, image_id in enumerate(retrieved_image_ids):
        LLM_response = llm_reranker.get_llm_confidence_score(query_image_name, image_id)
        scored_images.append({
            'id': image_id,
            'original_rank': rank + 1,
            'llm_score': int(LLM_response['confidence_score'])
        })
        query_reasons.append(LLM_response)

    # Sort by LLM score (descending), then by original rank (ascending) as a tie-breaker
    # TODO: We can further modify the sorting criteria by weighted LLM score and original rank
    reranked_images = sorted(scored_images, key=lambda x: (x['llm_score'], -x['original_rank']), reverse=True)
    return reranked_images, query_reasons

def save_reranked_results_to_file(output_file_path, all_reranked_data):
    """Saves the re-ranked image IDs to the specified output file."""
    try:
        with open(output_file_path, 'w') as f:
            for reranked_set in all_reranked_data:
                # Extracting just the IDs for the output file, maintaining the answer.txt format
                reranked_ids_only = [item['id'] for item in reranked_set]
                f.write('\t'.join(reranked_ids_only) + '\n')
        print(f"\nRe-ranked results saved to {output_file_path}")
    except IOError:
        print(f"Error: Could not write to output file '{output_file_path}'.")

# --- Main Execution ---
if __name__ == "__main__":
    answer_root_dir = os.path.join('src', 'results', '3')
    query_file_path = os.path.join('src','data','query_street_name.txt')
    answer_file_path = os.path.join(answer_root_dir, 'answer.txt')
    output_file_path = os.path.join(answer_root_dir, 're_ranked_answer.txt')  # Path for the output file

    LLM_MODEL = 'gemini'  # Change to 'ollama', 'gemini', or 'claude' as needed

    API_KEY = "AIzaSyDLhp4Bj32cs0YYdHzVWmAm_acFFY4Nf-o"
    llm_reranker_instance = LLMReRanker(mode=LLM_MODEL, api_key=API_KEY, data_root='../scratch/university-1652/University-1652/test/')

    query_names = read_query_names(query_file_path)
    initial_rankings = read_initial_rankings(answer_file_path)

    if not query_names or not initial_rankings:
        print("Could not read query or answer files. Exiting.")
    else:
        assert len(query_names) == len(initial_rankings), "Mismatch in number of queries and initial rankings."
        
        print(f"Processing {len(query_names)} queries...")

        all_reranked_results = []
        all_reasons = []
        for i in range(len(query_names)):
            current_query = query_names[i]
            current_retrieved_set = initial_rankings[i]
            
            print(f"\nRe-ranking for query: {current_query} ({i+1}/{len(query_names)})")
            print(f"  Initial retrieved IDs: {current_retrieved_set}")
            
            reranked_set, query_reasons = rerank_image_set(current_query, current_retrieved_set, llm_reranker_instance)
            all_reranked_results.append(reranked_set)
            all_reasons.append(query_reasons)
            
            # print(f"  Re-ranked IDs: {[img['id'] for img in reranked_set]}")
            # print(f"  Scores (LLM): {[f'{img['llm_score']:.2f}' for img in reranked_set]}")

        if all_reranked_results:
            save_reranked_results_to_file(output_file_path, all_reranked_results)

            # Optional: Print details of the first re-ranked set for inspection
            if all_reranked_results and all_reranked_results[0]:
                print("\n--- Details of the first re-ranked set ---")
                print(f"Query: {query_names[0]}")
                for item in all_reranked_results[0]:
                    print(f"  ID: {item['id']}, LLM Score: {item['llm_score']:.2f}, Original Rank: {item['original_rank']}")
                    
        with open(os.path.join(answer_root_dir, 'reasons.json'), 'w') as f:
            json.dump(all_reasons, f, indent=4)