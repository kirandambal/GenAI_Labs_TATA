# Add Azure OpenAI package
from openai import AzureOpenAI

azure_oai_endpoint = "https://azure2185454.openai.azure.com/"
azure_oai_key = "e667557368d248b2a5c70a50de90858f"
azure_oai_model = "Demo"

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=azure_oai_endpoint,
    api_key=azure_oai_key,
    api_version="2023-05-15"
)
email_text = ""

def text_input():
    global email_text
    while True:
        command = input("Enter email content (or type 'summary' or 'reply' to proceed): ")
        if command.lower() == 'summary':
            azopenai(email_text, "summary")
        elif command.lower() == 'reply':
            reply_type = input("Choose a reply type (professional, friendly, casual): ")
            azopenai(email_text, "reply", reply_type)
        else:
            email_text += " " + command

def text_output(response_text):
    print("\nGPT: \n" + response_text + "\n")
    print("\n##############################################################################")

def azopenai(text, command, reply_type=""):
    # Construct system message based on command
    system_message = """
    You are an AI assistant designed to help with email management, leveraging Azure OpenAI technologies.
    ---
    Functionality:
    ---
    1: Summarization:
    Generate a concise and coherent summary of the provided email content using OpenAI's GPT model.
    ---
    2: Email Reply Generation:
    Generate a reply to the provided email content. The reply can be 'professional', 'friendly', or 'casual' based on the user's choice.
    ---
    Provided email content: """ + text

    user_message = ""
    if command == "summary":
        user_message = "Generate a summary of the above email content."
    elif command == "reply":
        user_message = f"Generate a {reply_type} reply to the above email content."

    # Send request to Azure OpenAI model
    response = client.chat.completions.create(
        model=azure_oai_model,
        temperature=0.7,
        max_tokens=1200,
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    res = response.choices[0].message.content         
    text_output(res)

def main():
    try:
        print("Hello, Welcome :\n\n")
        text_input()

    except Exception as ex:
        print(ex)

if __name__ == '__main__': 
    main()