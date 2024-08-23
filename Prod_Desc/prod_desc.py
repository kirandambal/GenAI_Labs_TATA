import os
import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from openai import AzureOpenAI

# Azure Cognitive Services and Azure OpenAI settings
cog_endpoint = "https://azure57941564.cognitiveservices.azure.com/"
cog_key = "9962d6f4459241f086ad49026b8f038e"
azure_oai_endpoint = "https://azureoai1484.openai.azure.com/"
azure_oai_key = "757195576a0d4630b961c25b40ed3ce4"
azure_oai_model = "Demo"

# Initialize Azure Cognitive Services client
credential = CognitiveServicesCredentials(cog_key)
cv_client = ComputerVisionClient(cog_endpoint, credential)

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=azure_oai_endpoint, 
    api_key=azure_oai_key,  
    api_version="2023-09-01-preview"
)

def GetTextRead(image_file_path):
    print(f'Reading text from {image_file_path}\n')
    with open(image_file_path, mode="rb") as image_data:
        read_op = cv_client.read_in_stream(image_data, raw=True)

        operation_location = read_op.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        while True:
            read_results = cv_client.get_read_result(operation_id)
            if read_results.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                break
            time.sleep(1)
        
        ocr_text = []
        if read_results.status == OperationStatusCodes.succeeded:
            for page in read_results.analyze_result.read_results:
                for line in page.lines:
                    ocr_text.append(line.text)
    return ocr_text

def gptOpenAI(text): 
    try: 
        response = client.chat.completions.create(
            model=azure_oai_model,
            temperature=0.7,
            max_tokens=520,
            messages=[
                {"role": "system", "content": "Generate a concise product description based on the following extracted text from the product image:"},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content  
    except Exception as ex:
        print(ex)

def main():
    # Specify the path to the product image file
    image_path = 'bingo.jpg'

    # Extract text from the product image
    text_from_image = GetTextRead(image_path)
    extracted_text = " ".join(text_from_image)
    print("Extracted Text:\n", extracted_text)

    # Get the product description from Azure OpenAI
    description = gptOpenAI(extracted_text)
    print("\nProduct Description:\n", description)

if __name__ == "__main__":
    main()
