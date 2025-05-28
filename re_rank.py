import random
from openai import OpenAI
import os
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class LLMReRanker:
    def __init__(self, mode='ollama', api_key=None, data_root = '/work1/wshah/xzhang/data/university-1652/University-1652/test'):
        """
        Initializes the LLMReRanker.
        mode: str: The mode of operation. Options are 'ollama', 'gemini', or 'claude'.
        api_key: str: The API key for the LLM service. Required for 'gemini' and 'claude'.
        """
        if mode == 'ollama':
            self.client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama', # required, but unused
            )
            self.model = 'llama3'
        elif mode == 'gemini':
            self.client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
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
        This function would formulate a prompt, call the LLM API, and parse the response.
        For now, it simulates this process.

        Args:
            query_image_name (str): The file name of the query image.
            retrieved_image_id (str): The ID of the retrieved image.

        Returns:
            float: A confidence score between 0.0 and 1.0.
        """

        prompt = """
                I will give you a pair of images, a ground image and a satellite image. 
                Your job is to identify if the ground panorama is taken at the location of the satellite image. 
                You should first give a score between 0 and 1 to indicate how similar is the ground panorama 
                to the satellite image (for example, 0.8 very similar ground and satellite pairs that they share 
                similar street layout and the buildings look similar,0.0 not similar at all for the ground and satellite pair). 
                Moreover, please give the reason for the correspondence, for example, the building is the same in these two views, or the 
                road direction is the same, etc. If not, please also give the reason why it is not. Please organize your response in a json format with two fields:
                {
                    "confidence_score": <score between 0 and 1>,
                    "reason": "<reason for the correspondence or lack thereof>"
                }
                Please do not include any other text outside of the JSON format and the confidence score.
                """


        base64_query = encode_image(os.path.join(self.data_root, 'workshop_query_street', f'{query_image_name}'))
        base64_satellite = encode_image(os.path.join(self.data_root, 'workshop_gallery_satellite', f'{retrieved_image_id}.jpg'))

        # response = self.client.responses.create(
        #     model=self.model,
        #     input=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 { "type": "input_text", "text": "This is the ground image:" },
        #                 {
        #                     "type": "input_image",
        #                     "image_url": f"data:image/jpeg;base64,{base64_query}",
        #                 },
        #                 { "type": "input_text", "text": "This is the satellite image:" },
        #                 {
        #                     "type": "input_image",
        #                     "image_url": f"data:image/jpeg;base64,{base64_satellite}",
        #                 },
        #                 { "type": "input_text", "text": f"{prompt}" }
        #             ],
        #         }
        #     ],
        # )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
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
                        },
                        { "type": "text", "text": f"{prompt}" }
                    ],
                }
            ],
        )

        # llm_response_text = response.choices[0].message.content.strip()

        # if "Confidence Score:" in llm_response_text:
        #     score_str = llm_response_text.split("Confidence Score:")[1].strip()
        #     score = float(score_str)
        #     if 0.0 <= score <= 1.0:
        #         return score
        #     else:
        #         print(f"Warning: LLM score {score} out of range for {query_image_name}, {retrieved_image_id}. Defaulting to 0.0.")
        #         return 0.0
        # else:
        #     print(f"Warning: 'Confidence Score:' not in LLM response ('{llm_response_text}') for {query_image_name}, {retrieved_image_id}. Defaulting to 0.0.")
        #     return 0.0

        print(response.choices[0])
        exit(0)
        return 0.0  # Placeholder return value, replace with actual score extraction logic


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
    for rank, image_id in enumerate(retrieved_image_ids):
        confidence = llm_reranker.get_llm_confidence_score(query_image_name, image_id)
        scored_images.append({
            'id': image_id,
            'original_rank': rank + 1,
            'llm_score': confidence
        })

    # Sort by LLM score (descending), then by original rank (ascending) as a tie-breaker
    reranked_images = sorted(scored_images, key=lambda x: (x['llm_score'], -x['original_rank']), reverse=True)
    return reranked_images

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
    llm_reranker_instance = LLMReRanker(mode=LLM_MODEL, api_key=API_KEY)

    query_names = read_query_names(query_file_path)
    initial_rankings = read_initial_rankings(answer_file_path)

    if not query_names or not initial_rankings:
        print("Could not read query or answer files. Exiting.")
    else:
        assert len(query_names) == len(initial_rankings), "Mismatch in number of queries and initial rankings."
        
        print(f"Processing {len(query_names)} queries...")

        all_reranked_results = []
        for i in range(len(query_names)):
            current_query = query_names[i]
            current_retrieved_set = initial_rankings[i]
            
            print(f"\nRe-ranking for query: {current_query} ({i+1}/{len(query_names)})")
            print(f"  Initial retrieved IDs: {current_retrieved_set}")
            
            reranked_set = rerank_image_set(current_query, current_retrieved_set, llm_reranker_instance)
            all_reranked_results.append(reranked_set)
            
            print(f"  Re-ranked IDs: {[img['id'] for img in reranked_set]}")
            print(f"  Scores (LLM): {[f'{img['llm_score']:.2f}' for img in reranked_set]}")

        if all_reranked_results:
            save_reranked_results_to_file(output_file_path, all_reranked_results)

            # Optional: Print details of the first re-ranked set for inspection
            if all_reranked_results and all_reranked_results[0]:
                print("\n--- Details of the first re-ranked set ---")
                print(f"Query: {query_names[0]}")
                for item in all_reranked_results[0]:
                    print(f"  ID: {item['id']}, LLM Score: {item['llm_score']:.2f}, Original Rank: {item['original_rank']}")