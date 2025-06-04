import torch
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
import csv

# Setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

questions = [
    "What is the predominant environment in the image? (urban/suburban/rural/highway/industrial/natural/dense forestation/water body/mixed)",
    "What type of road layout is visible in the image? (grid pattern/winding roads/roundabout/dead-end streets/highway/none/mixed patterns)",
    "What specific environmental features are visible in the image? (Residential buildings / Commercial areas / Factories / Farms / Green spaces / Parks / Rivers / Lakes / Forests / Beaches / Cliffs / Hills / Open fields / Vacant lots / None / Other)",
    "What kind of distinct road features are present in the image? (none/simple intersections/complex junctions/overpasses/roundabouts/traffic circles)",
    "What types of buildings are most common in the image? (residential houses/apartment buildings/commercial buildings/industrial facilities/public buildings/mixed/no buildings/other)",
    "What is the condition of the vegetation in the image? (None/dense forests/parklands/sparse vegetation/agricultural fields/barren land/ornamental gardens)",
    "What distinctive features are present in the image? (None / Natural Landmarks / Historical Buildings / Modern Structures / Sporting Facilities / Water Bodies / Parks / Urban Art / Monuments / Industrial Facilities / Other)",
    "What is the architecture style of the buildings in the image ? (None/traditional/modern/industrial/mixed/historical)",
    "What type of transportation features can be seen in the image? (None/train tracks/airports/ports/tram lines/bus stations)",
    "What kind of large, open spaces are there in the picture? (None/fields/empty lots/forests/car parks/urban squares/golf course/public garden/playgrounds/sports field)",
    "What is the overall layout of the area observed in the image? (organized/disorganized/mixed/regular pattern/irregular pattern/none/chaotic)",
    "What are the unique patterns in roads or buildings in the image? (none/linear patterns/radial patterns/grid patterns/irregular patterns/circular patterns)",
    "What is the predominant color of the roofs in the image? (red/brown/grey/white/green/other/none/multi-colored)",
    "What is the predominant color of the roads in the image? (black/grey/red/yellow/other/none/multi-colored)",
    "What other notable color features are present in the image? (green areas/water bodies/colored buildings/sports fields/none/colorful gardens)",
    "What type of main road is visible in the image? (none/single-lane road/multi-lane road/highway/expressway)",
    "What road markings are present in the image? (None / Zebra crossings / Chevrons / White lines / Yellow lines / Double yellow lines / Arrows / Stop lines / Crosswalks / Bicycle lanes / Bus lanes / Hatched markings / Box junctions / School crossings / Speed limit markings / Other)",
    "What are the predominant colors of the road markings in the image? (White / Yellow / Red / Blue / Green / None / Other / Multi-colored)",
    "Are there any of the following road structures visible in the image? (None / Bridge / Underpass / Overpass / Tunnel / Flyover / Pedestrian crossing bridge / Roundabout / Highway interchange / Railway crossing / Other)",
    "How would you describe the orientation of the roads in the image? (Straight highway / Single road / Multiple parallel roads / Multiple roads converging / Multiple roads diverging / Intersection / Roundabout / Serpentine or winding road / T-junction / Crossroads / Forked road / Overpass/underpass systems / Cul-de-sac / One-way street / Pedestrian-only path / Bicycle lane / None / Other)",
    "What are the predominant types of surrounding vehicles in the image? (Cars / Trucks / Bicycles / Motorcycles / Public Transport / None)",
    "What is the directional layout of the road junction in the image? (none/left turn only / right turn only / both left and right turns / four-way intersection / roundabout / multiple direction options / complex multi-way junction / other)",
    "What is the width of the road? (None/narrow/medium/wide/multiple lanes/variable widths)",
    "Are there any traffic lights present along the road? (yes/no)",
    "Are there any billboard signs on the road indicating directions or destinations? (Yes / No)",
    "Is there a rest area or service station visible in the image? (yes/no)",
    "What type of service facility is visible in the image? (None/Petrol station / Supermarket / Restaurant / Hotel)",
    "Are any sports courts visible? (None/basketball/tennis/football)",
    "Does the road have a hard shoulder or emergency lane? (yes/no)",
    "Is there a pedestrian area like a sidewalk or footpath alongside the road? (yes/no)"
]

def generate_answers(input_folder, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        #header = ['Filename'] + [f'Question_{i+1}' for i in range(len(questions))]
        header = ['image_name'] + questions
        writer.writerow(header)

        # List all images in the input folder
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Process each image in the folder with a progress bar
        for filename in tqdm(image_files, desc="Processing images"):
            # Load and preprocess the image
            image_path = os.path.join(input_folder, filename)
            raw_image = Image.open(image_path).convert('RGB')
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            # Process each question individually
            answers = []
            for question in questions:
                processed_question = txt_processors["eval"](question)
                generative_answer = model.predict_answers(
                    samples={"image": image, "text_input": processed_question},
                    inference_method="generate"
                )
                answers.append(generative_answer[0])

            # Write the results to the CSV file
            writer.writerow([filename] + answers)

if __name__ == "__main__":
    input_folder = 'Path to input images'
    output_csv = 'Output_(satellite)_or_(query).csv'
    generate_answers(input_folder, output_csv)




    




from openai import OpenAI
from google import genai
import os
import base64
from google.genai import types
from pydantic import BaseModel
from enum import Enum
import json
import time


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
        self.client = text_encoder
        self.model = 'llama3'
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

        prompt = questions
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
        LLM_response['query_ground'] = query_image_name
        LLM_response['retrieved_satellite'] = retrieved_image_id

        # print(LLM_response)
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

    # 1 Sort by LLM score (descending), then by original rank (ascending) as a tie-breaker
    # reranked_images = sorted(scored_images, key=lambda x: (x['llm_score'], -x['original_rank']), reverse=True)
    
    # 2 (10 - llm_score) means less LLM score is better. Then add the original rank together. Less summed score means better rank.
    reranked_images = sorted(scored_images, key=lambda x: (10 - x['llm_score'] + x['original_rank']))
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

    API_KEY = "YOUR_API_KEY"
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
            while True:
                try:
                    time.sleep(60)  # Adding a delay to avoid hitting API rate limits
                    current_query = query_names[i]
                    current_retrieved_set = initial_rankings[i]
                    
                    print(f"\nRe-ranking for query: {current_query} ({i+1}/{len(query_names)})")
                    print(f"  Initial retrieved IDs: {current_retrieved_set}")
                    
                    reranked_set, query_reasons = rerank_image_set(current_query, current_retrieved_set, llm_reranker_instance)
                    all_reranked_results.append(reranked_set)
                    all_reasons.append(query_reasons)
                    break  # Exit the retry loop if successful
                except Exception as e:
                    print(f"Error during re-ranking for query '{current_query}': {e}")
                    print("Retrying after a short delay...")
                    time.sleep(10)
            
            # print(f"  Re-ranked IDs: {[img['id'] for img in reranked_set]}")
            # print(f"  Scores (LLM): {[f'{img['llm_score']:.2f}' for img in reranked_set]}")

        if all_reranked_results:
            save_reranked_results_to_file(output_file_path, all_reranked_results)

            # Optional: Print details of the first re-ranked set for inspection
            # if all_reranked_results and all_reranked_results[0]:
            #     print("\n--- Details of the first re-ranked set ---")
            #     print(f"Query: {query_names[0]}")
            #     for item in all_reranked_results[0]:
            #         print(f"  ID: {item['id']}, LLM Score: {item['llm_score']:.2f}, Original Rank: {item['original_rank']}")
                    
        with open(os.path.join(answer_root_dir, 'reasons.json'), 'w') as f:
            json.dump(all_reasons, f, indent=4)