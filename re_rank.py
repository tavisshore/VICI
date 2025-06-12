from openai import OpenAI
from google import genai
import os
import base64
from google.genai import types
from pydantic import BaseModel
from enum import Enum
import json
import time


# class Score(Enum):
#     ZERO = '0'
#     ONE = '1'
#     TWO = '2'
#     THREE = '3'
#     FOUR = '4'
#     FIVE = '5'
#     SIX = '6'
#     SEVEN = '7'
#     EIGHT = '8'
#     NINE = '9'
#     TEN = '10'

# class ResponseFormat(BaseModel):
#     confidence_score: Score
#     summary_ground_image: str
#     summary_satellite_image: str
#     reason: str

class ResponseFormat(BaseModel):
    ranking: list[int]
    summary_ground_image: str
    summary_satellite_images: list[str]
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
            

    def get_llm_confidence_score(self, query_image_name, retrieved_image_ids):
        """
        Gets a confidence score from the LLM for a query-retrieved image pair.

        Args:
            query_image_name (str): The file name of the query image.
            retrieved_image_ids (list[str]): The IDs of the retrieved images.

        Returns:
            float: A confidence score between 0.0 and 1.0.
        """

        prompt = """
                I will give you a ground image and 10 satellite images (the first satellite image is satellite 1 and the last one is satellite 10). 
                Your job is to identify which satellite is the location where the ground image was taken.
                You should first summarize the content of the ground image. Pay attention to the salient objects, such as streets, pedestrian ways, buildings, and other features that can help you determine if they are taken at the same location.
                Try to make the summary concise and informative.
                Do not pay attention to the time of the image, the weather conditions, or objects like cars, people, etc.
                Then you should take a look at each satellite image, summarize in a similar way as the ground image and find the corresponding objects between the satellite image and the given ground image. 
                Finally, you should rank these 10 satellite images from the most likely location to the least likely location. Your output should be a list of integers [1, 4, 7, 5, 8, 3, 9, 2, 6, 10], which means satellite 1 is the most probable one, and satellite image 10 is the least probable location.
                Lastly, for the most probable location (in the previous example is satellite image 1), give a concise reason for choosing it as the most likely location, giving the corresponding objects you find and identifying where exactly might be the ground camera location on this satellite image ( for example, you can say on the top right corner of the satellite image probably is the location where the ground image is taken at).
                Make your response concise.
                """
                
        json_format = """
                Please organize your response in a json format with following fields:
                
                ```
                {
                    "ranking": a list of integers, 1-10, representing the ranking of the satellite images from most likely to least likely location,
                    "Summary of ground iamge": "<summary of the ground image>",
                    "Summary of satellite image": "<summary of the satellite image>",
                    "reason": "<reason why choosing the most likely satellite image as the location of the ground image>"
                }
                ```
                Please only fill the fields in the above JSON template and do not include any other extra text inside and outside of the above JSON template.
                """

        # print(query_image_name)
        # print(retrieved_image_id)

        if 'gemini' not in self.model:
            # # For Ollama and Claude, we use the OpenAI client to create a response
            # base64_query = encode_image(os.path.join(self.data_root, 'workshop_query_street', f'{query_image_name}'))
            
            # base64_satellite = encode_image(os.path.join(self.data_root, 'workshop_gallery_satellite', f'{retrieved_image_id}.jpg'))
            
            # response = self.client.responses.create(
            #     model=self.model,
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": [
            #                 { "type": "text", "text": f"{prompt}+{json_format}" },
            #                 { "type": "text", "text": "This is the ground image:" },
            #                 {
            #                     "type": "image_url",
            #                     "image_url": {
            #                         "url": f"data:image/jpeg;base64,{base64_query}"
            #                     },
            #                 },
            #                 { "type": "text", "text": "This is the satellite image:" },
            #                 {
            #                     "type": "image_url",
            #                     "image_url": {
            #                         "url": f"data:image/jpeg;base64,{base64_satellite}"
            #                     },
            #                 }
            #             ],
            #         }
            #     ],
            # )
            
            # LLM_response = json.loads(response.choices[0].message.content)

            # For the other mdoels, we need to think how to get the output for top 10 images
            pass

        else:
            # For Gemini, we use the genai client to generate content
            with open(os.path.join(self.data_root, 'workshop_query_street', f'{query_image_name}'), 'rb') as f_q:
                query_bytes = f_q.read()

            requst_content = [
                f'{prompt}', 
                'This is the query ground image:',
                types.Part.from_bytes(
                    data=query_bytes, 
                    mime_type='image/jpeg'
                    )
                ]

            for index, retrieved_image_id in enumerate(retrieved_image_ids):
                with open(os.path.join(self.data_root, 'workshop_gallery_satellite', f'{retrieved_image_id}.jpg'), 'rb') as f_s:
                    satellite_bytes = f_s.read()

                requst_content.extend([
                    f'This is the satellite image {index + 1}:',
                    types.Part.from_bytes(
                        data=satellite_bytes,
                        mime_type='image/jpeg'
                    )
                ])

            response = self.client.models.generate_content(
                model=self.model,
                contents=requst_content,
                config=types.GenerateContentConfig(
                    temperature=0.0, # want to be more deterministic
                    thinking_config=types.ThinkingConfig(thinking_budget=1024),
                    response_mime_type='application/json',
                    response_schema=ResponseFormat,
                )
            )
            
            LLM_response = json.loads(response.text)
            
        LLM_response['query_ground'] = query_image_name
        LLM_response['retrieved_images'] = {retrieved_image_id:(index + 1) for index, retrieved_image_id in enumerate(retrieved_image_ids)}

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

def rerank_image_set(query_image_name, retrieved_image_ids, llm_reranker, keep_original_rank = False):
    """
    Re-ranks a single set of retrieved images for a query using LLM scores.
    """
    scored_images = []
    query_reasons = []

    if not keep_original_rank:
        LLM_response = llm_reranker.get_llm_confidence_score(query_image_name, retrieved_image_ids)
        
        for image_id, original_rank in LLM_response['retrieved_images'].items():
            scored_images.append({
                'id': image_id,
                'original_rank': original_rank,
                'llm_score': LLM_response['ranking'].index(original_rank) + 1,  # Convert to 1-based index
            })
            
        query_reasons.append(LLM_response)
    else:
        for rank, image_id in enumerate(retrieved_image_ids):
            scored_images.append({
                'id': image_id,
                'original_rank': rank + 1,
                'llm_score': rank + 1,  # Convert to 1-based index
            })
            
        query_reasons.append({'query_ground': query_image_name, 
                              'reason':'Skipped'})

    # 1 Sort by LLM score (descending), then by original rank (ascending) as a tie-breaker
    # reranked_images = sorted(scored_images, key=lambda x: (x['llm_score'], -x['original_rank']), reverse=True)
    
    # 2 (10 - llm_score) means less LLM score is better. Then add the original rank together. Less summed score means better rank.
    
    weighted_reranked_images = sorted(scored_images, key=lambda x: (x['llm_score'] + x['original_rank'], x['original_rank']))
    
    LLM_reranked_images = sorted(scored_images, key=lambda x: (x['llm_score'], x['original_rank']))
    
    # print(reranked_images)
    return weighted_reranked_images, LLM_reranked_images, query_reasons

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
    answer_root_dir = os.path.join('src', 'results', '0')
    query_file_path = os.path.join('src','data','query_street_name.txt')
    answer_file_path = os.path.join(answer_root_dir, 'answer.txt')
    weighted_output_file_path = os.path.join(answer_root_dir, 'weighted_re_ranked_answer.txt')  # Path for the weighted output file
    
    LLM_output_file_path = os.path.join(answer_root_dir, 'LLM_re_ranked_answer.txt')  # Path for the pure LLM output file

    LLM_MODEL = 'gemini'  # Change to 'ollama', 'gemini', or 'claude' as needed

    API_KEY = "GEMINI_API_KEY"  # Replace with your actual API key for Gemini
    llm_reranker_instance = LLMReRanker(mode=LLM_MODEL, api_key=API_KEY, data_root='../scratch/university-1652/University-1652/test/')

    query_names = read_query_names(query_file_path)
    initial_rankings = read_initial_rankings(answer_file_path)

    if not query_names or not initial_rankings:
        print("Could not read query or answer files. Exiting.")
    else:
        assert len(query_names) == len(initial_rankings), "Mismatch in number of queries and initial rankings."
        
        print(f"Processing {len(query_names)} queries...")

        all_weighted_reranked_results = []
        all_LLM_reranked_results = []
        all_reasons = []
        
        skipped_id = []
        for i in range(len(query_names)):
        # for i in range(446,449):
            exception_counter = 0
            while True:
                try:
                    current_query = query_names[i]
                    current_retrieved_set = initial_rankings[i]
                    
                    print(f"\nRe-ranking for query: {current_query} ({i+1}/{len(query_names)})")
                    print(f"  Initial retrieved IDs: {current_retrieved_set}")
                    
                    weighted_reranked_set, LLM_reranked_set, query_reasons = rerank_image_set(current_query, current_retrieved_set, llm_reranker_instance)
                    all_weighted_reranked_results.append(weighted_reranked_set)
                    all_LLM_reranked_results.append(LLM_reranked_set)
                    all_reasons.append(query_reasons)
                    break
                
                except Exception as e:
                    exception_counter += 1
                    print(f"Error processing query '{current_query}': {e}")
                    time.sleep(5)  # Wait before retrying
                    
                    # If larger than 3 times exception just abort this query
                    if exception_counter > 3:
                        
                        skipped_id.append(current_query)
                        
                        weighted_reranked_set, LLM_reranked_set, query_reasons = rerank_image_set(current_query, current_retrieved_set, llm_reranker_instance, keep_original_rank=True)
                        
                        all_weighted_reranked_results.append(weighted_reranked_set)
                        all_LLM_reranked_results.append(LLM_reranked_set)
                        all_reasons.append(query_reasons)
                        break

        if all_weighted_reranked_results:
            save_reranked_results_to_file(weighted_output_file_path, all_weighted_reranked_results)
            
        if all_LLM_reranked_results:
            save_reranked_results_to_file(LLM_output_file_path, all_LLM_reranked_results)
                    
        with open(os.path.join(answer_root_dir, 'reasons.json'), 'w') as f:
            json.dump(all_reasons, f, indent=4)
        
        print('number of skipped images: ', len(skipped_id))
        print('Skipped images: ', skipped_id)