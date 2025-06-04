from openai import OpenAI
import os
import base64
from pydantic import BaseModel
from enum import Enum
import json
import time
from tqdm import tqdm
from pathlib import Path
from google import genai
from google.genai import types
import csv
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

def write_list(fname, data):
    with open(fname, 'a') as f:
        for item in data:
            f.write(f'{item.replace('\n', '')} ')
        f.write('\n')

def read_existing(fname):
    queries = []
    with open(fname, 'r') as f:
        lines = f.readlines()
        for l in lines:
            val = l.split(' ')[0]
            queries.append(f'{val}.jpeg')
    return queries



def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class LLMReRanker:
    def __init__(self, mode='gemini', api_key=None, data_root = '/work1/wshah/xzhang/data/university-1652/University-1652/test'):
        """
        Initializes the LLMReRanker.
        mode: str: The mode of operation. Options are 'ollama', 'gemini', or 'claude'.
        api_key: str: The API key for the LLM service. Required for 'gemini' and 'claude'.
        data_root: str: The root directory for the dataset.
        """
        self.mode = mode
        if mode == 'ollama':
            self.client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama', # required, but unused
            )
            self.model = 'llama3'
        elif mode == 'gemini':
            self.client = genai.Client(
                api_key='AIzaSyBK4LqX_73ahBl9N2bvOpqfiXQTTHKYAvc',
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
            
    def get_llm_confidence_score(self, base64_image):
        prompt = """
                Answer the following questions by selecting a single option from the list, outputting only the answer and an appended ' - ', 
                Ensure all questions are answered:

                What is the predominant environment in the image? (urban/suburban/rural/highway/industrial/natural/dense forestation/water body/mixed),
                What type of road layout is visible in the image? (grid pattern/winding roads/roundabout/dead-end streets/highway/none/mixed patterns),
                What kind of distinct road features are present in the image? (none/simple intersections/complex junctions/overpasses/roundabouts/traffic circles),
                What types of buildings are most common in the image? (residential houses/apartment buildings/commercial buildings/industrial facilities/public buildings/mixed/no buildings/other),
                What is the condition of the vegetation in the image? (None/dense forests/parklands/sparse vegetation/agricultural fields/barren land/ornamental gardens),
                What distinctive features are present in the image? (None / Natural Landmarks / Historical Buildings / Modern Structures / Sporting Facilities / Water Bodies / Parks / Urban Art / Monuments / Industrial Facilities / Other),
                What is the architecture style of the buildings in the image ? (None/traditional/modern/industrial/mixed/historical),
                What type of transportation features can be seen in the image? (None/train tracks/airports/ports/tram lines/bus stations),
                What kind of large, open spaces are there in the picture? (None/fields/empty lots/forests/car parks/urban squares/golf course/public garden/playgrounds/sports field),
                What is the overall layout of the area observed in the image? (organized/disorganized/mixed/regular pattern/irregular pattern/none/chaotic),
                What are the unique patterns in roads or buildings in the image? (none/linear patterns/radial patterns/grid patterns/irregular patterns/circular patterns),
                What is the predominant color of the roofs in the image? (red/brown/grey/white/green/other/none/multi-colored),
                What is the predominant color of the roads in the image? (black/grey/red/yellow/other/none/multi-colored),
                What other notable color features are present in the image? (green areas/water bodies/colored buildings/sports fields/none/colorful gardens),
                What type of main road is visible in the image? (none/single-lane road/multi-lane road/highway/expressway),
                What road markings are present in the image? (None / Zebra crossings / Chevrons / White lines / Yellow lines / Double yellow lines / Arrows / Stop lines / Crosswalks / Bicycle lanes / Bus lanes / Hatched markings / Box junctions / School crossings / Speed limit markings / Other),
                What are the predominant colors of the road markings in the image? (White / Yellow / Red / Blue / Green / None / Other / Multi-colored),
                Are there any of the following road structures visible in the image? (None / Bridge / Underpass / Overpass / Tunnel / Flyover / Pedestrian crossing bridge / Roundabout / Highway interchange / Railway crossing / Other),
                How would you describe the orientation of the roads in the image? (Straight highway / Single road / Multiple parallel roads / Multiple roads converging / Multiple roads diverging / Intersection / Roundabout / Serpentine or winding road / T-junction / Crossroads / Forked road / Overpass/underpass systems / Cul-de-sac / One-way street / Pedestrian-only path / Bicycle lane / None / Other),
                What are the predominant types of parked vehicles in the image? (Cars / Trucks / Bicycles / Motorcycles / Public Transport / None),
                What is the directional layout of the road junction in the image? (none/left turn only / right turn only / both left and right turns / four-way intersection / roundabout / multiple direction options / complex multi-way junction / other),
                What is the width of the road? (None/narrow/medium/wide/multiple lanes/variable widths),
                Are there any traffic lights present along the road? (yes/no),
                Are there any billboard signs on the road indicating directions or destinations? (Yes / No),
                Is there a rest area or service station visible in the image? (yes/no),
                What type of service facility is visible in the image? (None/Petrol station / Supermarket / Restaurant / Hotel),
                Are any sports courts visible? (None/basketball/tennis/football),
                Does the road have a hard shoulder or emergency lane? (yes/no),
                Is there a pedestrian area like a sidewalk or footpath alongside the road? (yes/no)
                """

        # Removed
        # What specific environmental features are visible in the image? (Residential buildings / Commercial areas / Factories / Farms / Green spaces / Parks / Rivers / Lakes / Forests / Beaches / Cliffs / Hills / Open fields / Vacant lots / None / Other),
                
                
        if self.mode == 'gemini':
            # For Gemini, we use the genai client to generate content

            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(
                              data=base64_image,
                              mime_type='image/jpeg',
                            ),
                            prompt
                          ],
                # config=types.GenerateContentConfig(
                #     temperature=0.0, # want to be more deterministic
                #     thinking_config=types.ThinkingConfig(thinking_budget=512),
                #     response_mime_type='application/json',
                #     response_schema=ResponseFormat,
                # )
            )

            output = response.candidates[0].content.parts[0].text
        else:
            response = self.client.chat.completions.create(
                # model="gpt-4o",
                model='claude-3-7-sonnet-20250219',
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )

            output = response.choices[0].message.content
        output = output.split(' - ')
        return output 


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

    # First get descriptions of query image
    base64_image = encode_image(os.path.join('/scratch/datasets/University/test/', 'workshop_query_street', f'{query_image_name}.jpeg'))
    query_response = llm_reranker.get_llm_confidence_score(base64_image)

    for rank, image_id in enumerate(retrieved_image_ids):
        base64_image = encode_image(os.path.join('/scratch/datasets/University/test/', 'workshop_gallery_satellite', f'{image_id}.jpeg'))
        ref_response = llm_reranker.get_llm_confidence_score(base64_image)

        score = 0
        for item_idx in range(len(query_response)):
            if query_response[item_idx] == ref_response[item_idx]:
                score += 1
            
        scored_images.append({'id': image_id, 'original_rank': rank + 1, 'score': score})
    
    reranked_images = sorted(scored_images, key=lambda x: x['score'], reverse=True)

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
    answer_root_dir = os.path.join('results')
    query_file_path = os.path.join('src','data','query_street_name.txt')
    answer_file_path = os.path.join(answer_root_dir, 'answer.txt')
    output_file_path = os.path.join(answer_root_dir, 're_ranked_answer.txt')  # Path for the output file

    LLM_MODEL = 'gemini'  # Change to 'ollama', 'gemini', or 'claude' as needed
    API_KEY = "YOUR_API_KEY"
    llm_reranker_instance = LLMReRanker(mode=LLM_MODEL, api_key=API_KEY, data_root='/scratch/university-1652/University-1652/test/')

    # First get the answers for all query and reference images - this will save a lot of querying
    query_names = read_query_names(query_file_path)

    existing = read_existing('query.txt')
    for x in existing:
        if x in query_names:
            query_names.remove(x)


    for q in tqdm(query_names, total=len(query_names)):
        success = False
        attempts = 0

        base64_image = open(os.path.join('/scratch/datasets/University/test/', 'workshop_query_street', f'{q}'), 'rb')
        image_bytes = base64_image.read()
        
        while not success and attempts < 5:
            query_response = llm_reranker_instance.get_llm_confidence_score(image_bytes)
            query_response = [x.lower() for x in query_response]
            query_response = query_response[0].split('\n')
            query_response = [f'{x} ' for x in query_response]

            if len(query_response) == 29:
                success = True
            else:
                print(f'retry {attempts}')
                attempts += 1

        if success:
            query_response.insert(0, q.split('.')[0])
            write_list('query.txt', query_response)









    ref_names = list(Path('/scratch/datasets/University/test/workshop_gallery_satellite').glob('*.jpeg'))

    existing = read_existing('ref.txt')
    for x in existing:
        if x in ref_names:
            ref_names.remove(x)

    for q in tqdm(ref_names, total=len(ref_names)):
        success = False
        attempts = 0

        base64_image = open(os.path.join('/scratch/datasets/University/test/', 'workshop_gallery_satellite', f'{q}'), 'rb')
        image_bytes = base64_image.read()
        
        while not success and attempts < 5:
            query_response = llm_reranker_instance.get_llm_confidence_score(image_bytes)
            query_response = [x.lower() for x in query_response]
            query_response = query_response[0].split('\n')
            query_response = [f'{x} ' for x in query_response]

            if len(query_response) == 29:
                success = True
            else:
                print(f'retry {attempts}')
                attempts += 1

        if success:
            query_response.insert(0, q.split('.')[0])
            write_list('ref.txt', query_response)
