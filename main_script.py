import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from pymongo import MongoClient
import json
import time

# Configure the Generative AI API
genai.configure(api_key="AIzaSyDlqbyGCrymD7m6bQ4FqD_HZR6k81SeS7w")

# Define the model configuration
generation_config = {
    "temperature": 2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 50000,
    "response_mime_type": "application/json",
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB connection string
db = client["Biomarkers"]  # Database name
collection = db["Protein Biomarkers"]  # Collection name

# Define the biomarker extraction prompt
def create_prompt(full_text):
    return f"""
RAG Prompt for Biomarker Data Extraction
Context:
You are an advanced AI system designed to analyze and extract specific information about protein biomarkers from research articles. You will process scientific content and extract details from research articles, including abstracts, figures, tables, and main body text. The data should be complete, accurate, and structured for database storage. Use the provided context to understand the task requirements.

Task:
Extract and organize the following information from the entire research article:
Biomarker Information:
Protein Name(s): Extract protein biomarkers mentioned in the document.
UniProt ID: Retrieve the UniProt ID(s) for the identified proteins.
Protein Sequence: Retrieve the full protein sequence from UniProt.
Isoforms: Record any specific isoforms mentioned (e.g., shorter, longer, or variants), along with their sequences.
Study and Source Information:
Disease Name(s): Identify the disease(s) the biomarker is associated with.
Source Material: Extract the biological source of the biomarker (e.g., serum, blood, saliva, urine, amniotic fluid, cerebrospinal fluid, etc.).
Organism: Always verify the organism as Homo sapiens (humans).
Technique Used: Extract the technique used to identify the biomarker (e.g., MS/MS or other analytical methods).
PubMed ID (PMID): Record the unique PubMed ID of the article.
Additional Information:
Alternative Protein Names: Retrieve alternative names or aliases for the protein(s) from UniProt.
Verification: Ensure that the identified biomarkers refer to proteins (not genes).
Validation and Special Cases:
Isoform Implication: If an isoform of a protein is specifically implicated in a disease, record its details separately, including sequence and disease association.
Accurate Categorization: Ensure the information aligns with proteins and not other biological molecules.
Output Format:
Return the information in JSON format for easy storage and processing. Ensure all fields are present even if the value is null.

Full Text of the Article:
{full_text}
"""

# Process text files with the generative AI model
def process_texts_with_chatbot(data_folder, output_json_folder):
    results = []
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as text_file:
                full_text = text_file.read()

            prompt = create_prompt(full_text)
            chat_session = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
            time.sleep(5)
            response = chat_session.send_message(prompt)
            time.sleep(5)
            results.append({"file_name": file_name, "response": response.text})

            # Save response to JSON file
            try:
                # Parse the response (which is a list of dictionaries) and extract the first item
                json_response = json.loads(response.text)
                
                # Check if the response is a list and extract the dictionary
                if isinstance(json_response, list) and len(json_response) == 1:
                    json_response = json_response[0]  # Extract the first dictionary from the list
                elif not isinstance(json_response, dict):
                    raise ValueError(f"Expected a dictionary but got {type(json_response).__name__}")

                # Save the response to a JSON file
                json_file_name = os.path.join(output_json_folder, file_name.replace(".txt", ".json"))
                with open(json_file_name, "w", encoding="utf-8") as json_file:
                    json.dump(json_response, json_file, indent=4, ensure_ascii=False)
                print(f"Saved response to {json_file_name}.")

                # Insert the response into MongoDB (as a dictionary)
                collection.insert_one(json_response)
                print(f"Inserted response for {file_name} into MongoDB.")

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for {file_name}: {e}")
                print(f"Raw response:\n{response.text}")
            except ValueError as e:
                print(f"Invalid JSON structure for {file_name}: {e}")
            except Exception as e:
                print(f"Error saving to MongoDB for {file_name}: {e}")

    return results

# Download PubMed HTML files and convert to text
def download_pubmed_html(pubmed_links_file, data_folder, output_json_folder, batch_size=1):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Read PubMed links from the file
    with open(pubmed_links_file, "r") as file:
        pubmed_links = [line.strip() for line in file if line.strip()]

    for i in range(0, len(pubmed_links), batch_size):
        batch_links = pubmed_links[i:i+batch_size]
        for idx, pubmed_link in enumerate(batch_links):
            try:
                response = requests.get(pubmed_link)
                if response.status_code == 200:
                    file_name = os.path.join(data_folder, f"article_{i + idx + 1}.html")
                    with open(file_name, "w", encoding="utf-8") as html_file:
                        html_file.write(response.text)
                else:
                    print(f"Failed to download {pubmed_link}: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {pubmed_link}: {e}")

        # Convert HTML files to text
        extract_text_from_html(data_folder)

        # Process text files and save responses
        process_texts_with_chatbot(data_folder, output_json_folder)

        # Delete processed files
        cleanup_data_folder(data_folder)

# Extract text from HTML files
def extract_text_from_html(data_folder):
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".html"):
            file_path = os.path.join(data_folder, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as html_file:
                    soup = BeautifulSoup(html_file, "html.parser")
                    text_content = soup.get_text(separator=" ", strip=True)

                text_file_path = os.path.splitext(file_path)[0] + ".txt"
                with open(text_file_path, "w", encoding="utf-8") as text_file:
                    text_file.write(text_content)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Cleanup data folder after processing
def cleanup_data_folder(data_folder):
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".txt") or file_name.endswith(".html"):
            file_path = os.path.join(data_folder, file_name)
            os.remove(file_path)

if __name__ == "__main__":
    pubmed_links_file = "pubmed_links.txt"  # File containing PubMed article links
    data_folder = "./data"  # Folder to store HTML files and extracted text
    output_json_folder = "./json_responses"  # Folder to save JSON files

    # Step 1: Download, process, and save responses
    download_pubmed_html(pubmed_links_file, data_folder, output_json_folder)

