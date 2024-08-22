import azure.cognitiveservices.speech as speech_sdk
from openai import AzureOpenAI

ai_key = "b3a46553536c4770bf9f793724875f06"
ai_region = "eastus"
azure_oai_endpoint = "https://azureoai1484.openai.azure.com/"
azure_oai_key = "c9e9cfc7702d413d85303f452d9a1980"
azure_oai_model = "Demo"

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=azure_oai_endpoint, 
    api_key=azure_oai_key,  
    api_version="2023-09-01-preview"
)

basetext = "Finance:"

# Speech to Text
def speechrec():
    command = ''

    # Configure speech service
    speech_config = speech_sdk.SpeechConfig(subscription=ai_key, region=ai_region)
    print('Ready to use speech service in:', speech_config.region)

    # Configure speech recognition
    audio_config = speech_sdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config)

    while command.lower() != 'quit.':
        print('User :')    
        # Process speech input
        speech = speech_recognizer.recognize_once_async().get()
        if speech.reason == speech_sdk.ResultReason.RecognizedSpeech:
            command = speech.text
            print(command)
        else:
            print(speech.reason)
            if speech.reason == speech_sdk.ResultReason.Canceled:
                cancellation = speech.cancellation_details
                print(cancellation.reason)
                print(cancellation.error_details)   
        azopenai(command)

# Text to Speech
def speechsys(response_text):
    # Configure speech synthesis
    speech_config.speech_synthesis_voice_name = 'en-GB-LibbyNeural'
    speech_synthesizer = speech_sdk.SpeechSynthesizer(speech_config)

    # Synthesize spoken output
    speak = speech_synthesizer.speak_text_async(response_text).get()
    if speak.reason != speech_sdk.ResultReason.SynthesizingAudioCompleted:
        print(speak.reason)

def azopenai(text):
    global basetext 
    basetext += ", " + text
    # Send request to Azure OpenAI model
    response = client.chat.completions.create(
        model=azure_oai_model,
        temperature=0.7,
        max_tokens=120,
        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful AI assistant designed for the banking industry, leveraging Azure Speech and OpenAI technologies to assist financial advisors and bankers by transcribing and summarizing client information accurately and efficiently.
                ---
                Functionality:
                ---
                1: Speech Recognition and Transcription
                Use Azure Speech Service to convert spoken information from client meetings, financial advisories, and internal discussions into text. This ensures high accuracy and captures financial terminologies effectively.
                ---
                2: Information Storage:
                Once the financial advisor provides information specific to a client or meeting, securely store this data and confirm with the response, "Remembered."
                ---
                3: Summarization:
                When requested, generate a concise and coherent summary of the stored client information or meeting notes using OpenAI's GPT model. The summary should be clear, relevant, and include all critical details discussed.
                ---
                Provided reference content: """ + basetext
            },
            {"role": "user", "content": text}
        ]
    )
    res = response.choices[0].message.content   
    print("\n\nGPT : \n" + res + "\n")
    print("\n##############################################################################")
    speechsys(res)

def main():
    try:
        global speech_config
        # Configure speech service
        speech_config = speech_sdk.SpeechConfig(subscription=ai_key, region=ai_region)
        print("Hello, Welcome to Speech to speech GPT model:\n\n")
        speechrec()

    except Exception as ex:
        print(ex)

if __name__ == '__main__': 
    main()
