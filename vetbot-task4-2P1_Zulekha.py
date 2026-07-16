import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load the environment variables from your secure .env file
load_dotenv()
print(os.getenv('Gemini-API-Key'))
print(os.getenv("MISTRAL_API_KEY"))

# Initialize the model using the updated 2026 standard model
model = init_chat_model(
    "gemini-3.5-flash",       # <-- Updated from 2.5 to 3.5
    model_provider="google_genai",
    temperature = 0.2         # decides how much predictable and repeated the model is, 0 always same, 1 always new, both are good for diff cases, it can range from 0-1
)
model1 = init_chat_model(
    "mistral-large-latest", # Or "mistral-small-latest"
    model_provider="mistralai",
    temperature=0.2
)

# #just to check that the mdel is working and which parameters, i'll comment it since no need after checking
# print("Lang chain model: \n")
# print(model, '\n==========================\n')

behavior = SystemMessage(content= ("You're an animal's expert, that helps us to take care of our pets well with vet advices, though no extra long answers"))  #sets up how the AI behaves
messages = [behavior]       # look them in one var to be called together

# Invoke the model messages to answer (to set up behavior)
print("\nClean Text Output: \n")
while True:
    inp = input("You: ")

    if inp.strip().lower() == 'exit':
        print('Hope this helped you, good bye!')
        break
    if not inp.strip():
        continue

    question = HumanMessage(content=(inp))
    messages.append(question)
    try:
        response = model.invoke(messages)
    except ChatGoogleGenerativeAIError as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            response = model1.invoke(messages)
        else:
            raise e
    
    if isinstance(response.content, list):      #if the var response is in a list
        clean_text = ""                         # empty var for response i it needs cleaning
        for block in response.content:          #each block in the og response
            if isinstance(block, dict) and block.get("type") == "text":     #if the blocks are dict and have words in form of text (notice dict uses .get('type'))
                clean_text += block.get("text", "")                         #then print the text type only 

            else:
                clean_text += str(block)        #Fallback if it's a list of standard string blocks then print it
        print(clean_text)

    else:
        clean_text = response.content
        print(clean_text)                 # If it's a standard plain string, print it directly
    responded = AIMessage(content=(clean_text))
    messages.append(responded)