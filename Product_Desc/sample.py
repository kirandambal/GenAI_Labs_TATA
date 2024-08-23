from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

# Azure Cognitive Services and Azure OpenAI settings
cog_endpoint = "Azure AI Service Endpoint"
cog_key = "Azure AI service Key"

# Initialize Azure Cognitive Services client
credential = CognitiveServicesCredentials(cog_key)
cv_client = ComputerVisionClient(cog_endpoint, credential)

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

