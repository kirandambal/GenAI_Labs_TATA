import os
import time
import fitz  # PyMuPDF library for working with PDF files
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speech_sdk

# Azure Cognitive Services and Azure OpenAI settings
cog_endpoint = "https://azure57941564.cognitiveservices.azure.com/"
cog_key = "9962d6f4459241f086ad49026b8f038e"
cog_region = "eastus"
azure_oai_endpoint = "https://azureoai1484.openai.azure.com/"
azure_oai_key = "757195576a0d4630b961c25b40ed3ce4"
azure_oai_model = "Demo"

# Initialize Azure Cognitive Services client
credential = CognitiveServicesCredentials(cog_key)
cv_client = ComputerVisionClient(cog_endpoint, credential)
speech_config = speech_sdk.SpeechConfig(subscription=cog_key, region=cog_region)

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=azure_oai_endpoint, 
    api_key=azure_oai_key,  
    api_version="2023-09-01-preview"
)

def GetTextRead(pdf_file_path):
    print('Reading text in {}\n'.format(pdf_file_path))
    # Use Read API to read text in PDF
    with open(pdf_file_path, mode="rb") as pdf_data:
        read_op = cv_client.read_in_stream(pdf_data, raw=True)

        # Get the async operation ID so we can check for the results
        operation_location = read_op.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        # Wait for the asynchronous operation to complete
        while True:
            read_results = cv_client.get_read_result(operation_id)
            if read_results.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                break
            time.sleep(1)
        
        ocr_text = []
        # If the operation was successful, process the text line by line
        if read_results.status == OperationStatusCodes.succeeded:
            for page in read_results.analyze_result.read_results:
                for line in page.lines:
                    ocr_text.append(line.text)
    return ocr_text

def gptOpenAI(text, prompt): 
    try: 
        # Send request to Azure OpenAI model
        response = client.chat.completions.create(
            model=azure_oai_model,
            temperature=0.7,
            max_tokens=520,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content  
    except Exception as ex:
        print(ex)


def speech(text):
    response_text = text
    # Configure speech synthesis
    speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
    speech_synthesizer = speech_sdk.SpeechSynthesizer(speech_config)    
    # Synthesize spoken output
    speech_synthesizer.speak_text_async(response_text).get()

def main():
    # Path to the training outline PDF file
    pdf_path = "AI- 900.pdf"
    
    # Extract text from the PDF
    text_from_pdf = GetTextRead(pdf_path)
    text = " ".join(text_from_pdf)
    


    # Generate a 4-5 line summary of the training outline
    summary_prompt = "Generate a concise summary (4-5 lines) of the training outline:"
    summary = gptOpenAI(text, summary_prompt)
    print("\nSummary of the Training Outline:\n")
    print(summary)
    
    while True:
        choice = input("\nWould you like to generate an explanation or an assessment? (Type 'explanation', 'assessment', or 'quit'):\n").strip().lower()
        
        if choice == "quit":
            break
        elif choice == "explanation":
            explanation_prompt = "Generate a one-line explanation for each module in the training outline:"
            explanation = gptOpenAI(text, explanation_prompt)
            print("\nExplanation of the Training Outline:\n")
            print(explanation)
        elif choice == "assessment":
            assessment_prompt = "Generate 5 basic multiple-choice questions (MCQs) with answers referring to the topics included in the training outline contents:"
            assessment = gptOpenAI(text, assessment_prompt)
            print("\nAssessment for the Training Outline:\n")
            print(assessment)
        else:
            print("Invalid choice. Please type 'explanation', 'assessment', or 'quit'.")

if __name__ == "__main__":
    main()
