import random
from openai import OpenAI

class LLMReRanker:
    def __init__(self, openai_api_key=None):
        """
        Initializes the LLMReRanker.
        If an openai_api_key is provided, it would set up the OpenAI client.
        """
        self.client = None
        if openai_api_key:
            try:
                self.client = OpenAI(api_key=openai_api_key)
                print("OpenAI client initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
        else:
            print("LLMReRanker initialized without OpenAI API key (simulation mode).")
            pass

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
        prompt = f"""
        Query Image Reference: "{query_image_name}"
        Retrieved Image Reference: "{retrieved_image_id}"

        Considering the query image reference and the retrieved image reference,
        how relevant is the retrieved image to the query image?
        Please provide a confidence score indicating relevance on a scale of 0.0 (not at all relevant)
        to 1.0 (highly relevant).

        Your response should be formatted ONLY as:
        Confidence Score: <score_value>
        
        Example:
        Confidence Score: 0.85
        """

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Or your preferred model
                    messages=[
                        {"role": "system", "content": "You are an AI assistant that evaluates the relevance of a retrieved image to a query image and provides a confidence score."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=20,
                    temperature=0.2
                )
                llm_response_text = response.choices[0].message.content.strip()
                # print(f"LLM Raw Response: {llm_response_text}")
                # Parse the score
                if "Confidence Score:" in llm_response_text:
                    try:
                        score_str = llm_response_text.split("Confidence Score:")[1].strip()
                        score = float(score_str)
                        if 0.0 <= score <= 1.0:
                            return score
                        else:
                            print(f"Warning: LLM score {score} out of range for {query_image_name}, {retrieved_image_id}. Defaulting to 0.0.")
                            return 0.0
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing score from LLM response ('{llm_response_text}') for {query_image_name}, {retrieved_image_id}: {e}. Defaulting to 0.0.")
                        return 0.0
                else:
                    print(f"Warning: 'Confidence Score:' not in LLM response ('{llm_response_text}') for {query_image_name}, {retrieved_image_id}. Defaulting to 0.0.")
                    return 0.0
            except Exception as e:
                print(f"Error calling OpenAI API for {query_image_name}, {retrieved_image_id}: {e}. Defaulting to 0.0.")
                return 0.0
            # pass # Keep simulation if client part is commented out

        # Simulate LLM score if self.client is None or API call fails/is commented out
        simulated_score = round(random.uniform(0.1, 1.0), 2)
        # print(f"Simulated Confidence Score: {simulated_score}")
        return simulated_score


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
    query_file_path = "query_street_name.txt"  # Path to your query_street_name.txt file [cite: 2]
    answer_file_path = "answer.txt"            # Path to your answer.txt file [cite: 1]
    output_file_path = "reranked_answers.txt"  # Path for the output file

    # --- IMPORTANT ---
    # To use the actual OpenAI API, you need to provide your API key.
    # For example:
    OPENAI_API_KEY = "your_openai_api_key_here"
    llm_reranker_instance = LLMReRanker(openai_api_key=OPENAI_API_KEY)
    
    # For this framework, we'll run in simulation mode (no actual API calls).
    # llm_reranker_instance = LLMReRanker()

    query_names = read_query_names(query_file_path)
    initial_rankings = read_initial_rankings(answer_file_path)

    if not query_names or not initial_rankings:
        print("Could not read query or answer files. Exiting.")
    else:
        # Handle potential mismatch in the number of lines between query and answer files
        # num_queries = len(query_names)
        # num_answer_sets = len(initial_rankings)

        # if num_queries != num_answer_sets:
        #     print(f"Warning: Mismatch in line counts. Queries: {num_queries}, Answers: {num_answer_sets}.")
        #     print(f"Processing up to the minimum common count: {min(num_queries, num_answer_sets)}.")
        #     min_count = min(num_queries, num_answer_sets)
        #     query_names = query_names[:min_count]
        #     initial_rankings = initial_rankings[:min_count]
        
        assert len(query_names) == len(initial_rankings), "Mismatch in number of queries and initial rankings."
        
        print(f"Processing {len(query_names)} queries...")

        all_reranked_results = []
        for i in range(len(query_names)):
            current_query = query_names[i]
            current_retrieved_set = initial_rankings[i]

            if not current_retrieved_set:
                print(f"No retrieved images for query '{current_query}'. Skipping re-ranking for this query.")
                all_reranked_results.append([]) # Add empty list to maintain structure
                continue
            
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