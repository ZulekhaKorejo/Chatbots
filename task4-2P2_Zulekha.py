#importing
import os
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
#from langchain_huggingface import HuggingFaceEndpoint ##reached limit
from pydantic import BaseModel, Field, ValidationError

#load the API
load_dotenv()
print(os.getenv("Gemini_API_Key", "MISTRAL_API_KEY"))

#init the models
gemini = init_chat_model(
        "gemini-3.5-flash",
    model_provider="google_genai",
    temperature = 0.1
)
mistral = init_chat_model(
    "mistral-large-latest",
    model_provider="mistralai",
    temperature=0.1
)
# switch between them if any is up

#now to make it in a good defined shape/schema we make a class
class extraction(BaseModel):
    purpose : str = Field(description="summary of what's in the email and the request")
    request_type : str = Field(description="rent, apartment issue, late fee excuse")
    amounts : float
    names : str #= Field(description="senders, reciever"), would've added thi to make it good but this good instructions will make all scores = 10
    dates : str
    reasoning : str

#now we init our models to be in structured form
gem_str = gemini.with_structured_output(extraction)
mis_str = mistral.with_structured_output(extraction)

#also since i'll switch them midway if ended i'm making a client func, all in clean format already
def model(emails: str):
    retries = 3
    for attempt in range(retries):
        try:
            result = gem_str.invoke(emails)
            return result
        
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                result = mis_str.invoke(emails)
                return result
            
            # If it's a validation error, we catch it here
            elif isinstance(e, ValidationError):
                print(f"validation error: retrying {attempt+1}")
                time.sleep(0.1)
            
            else:
                print(f"failed, Error: {e}")
                
    return None


#the email i'll extract in loop
test_emails = [
    "Date: 16-07-2026. Hi, I sent the $1500 rent to landlady Mrs. Higgins yesterday from checking 4412. - Marc Spencer",
    "Sent my half of eight hundred dollars to Dave on July 1st. Account ending 9921. - Sarah J.",
    "Is my $1200 rent payment cleared? Sent it 2 days ago to landlord Tony. Checking last digits 1102. Thanks, Peter Parker",
    "Hey! Just confirming the security deposit of $2500 was sent to Apex Rentals on 12-07-2026. - Bruce Wayne (Acct: 8831)",
    "Dear Jonathan, my automatic transfer of $1350 to Mrs. Gable scheduled for the 5th of this month failed. Acct #6721. - Monica G.",
    "mistakenly sent an extra $450 to Apex Rentals on June 15th from savings ending 2311. Please credit next month. - Clark Kent",
    "Hi, paid the $15 convenience fee on July 14th. My account ends in 7741. - Barry Allen",
    "My rent of nine hundred dollars was sent to Jessica Shulz on the 3rd of last month. Acct 5512. - Wade Wilson",
    "I noticed a charge of $20.50 on 30-06-2026. Can you refund it to account 4403? - Clara Bow",
    "Date: 16-07-2026. Rent of 1700 paid to Jonathan Reyes on July 14th. From checking 4321. - Mark Davis"
]

#deciding how it'll behave with each iteration
# The rules for the AI
prompts = {
    "1_List_Maker": "Extract: purpose, Request_Type, Amount, Names, Dates. Explain the math in 3 simple points from {email}",
    
    "2_Thinker": """Looking at {email}, think step-by-step:
        1. Is this rent?
        2. Benchmark Math: Use $850 as the fixed monthly rent. Calculate the difference: (Payment Amount - $850). Explicitly define the result as an 'Overpayment of X' or a 'Shortfall of X'.
        3. What is the purpose of the email?
        4. Is it late? Provide 3 points of reasoning (consider due dates, payment date, and grace periods).""",
    
    "3_Robot_Auditor": """extract from {email}, Rules:
    1. If not rent, mark as IGNORE.
    2. Math: Compare amount to $850.
    3. Reasoning must be 3 points:
       - 'Payment Data': [Names and amounts]
       - 'Timeline Audit': [Date and late status]
       - 'Discrepancy Log': [The math calculation]
    4. Example output:
    "purpose": "Short payment notification from John Doe regarding July rent.",
    "request_type": "rent",
    "amounts": 700,
    "names": "Sender: John Doe, Receiver: Property Manager",
    "dates": "2026-07-03",
    "reasoning":
    1. Payment Data: John Doe paid $700.
    2. Timeline Audit: Received on July 3rd; payment is 2 days late (Due 1st).
    3. Discrepancy Log: Expected $850, Paid $700, Shortfall of $150."""
}

#making a scoring function to make a table
def score(answer):
    # required = ['title', 'request_type', 'amounts', 'names', 'dates', 'reasoning']
    # for key in answer:
    #     return all(key in required) #checks each key in required with each in answer if they exist or not
    #this was a good approach but it didn't check if the extraction boxesa are empty, filled or wtvr, as long as they key exists
    sc =10
    if answer is None or isinstance(answer, str):
        return 0
    if len(answer.purpose) <=7:
        sc -= 2
    if answer.amounts <= 0:
        sc -= 1
    if len(answer.names)<2:
        sc -= 1
    if len(answer.reasoning) <= 10:
        sc -= 1
    if "Payment Data" not in answer.reasoning:
        sc -= 1
    if "Timeline Audit" not in answer.reasoning:
        sc -= 1
    if "Discrepancy Log" not in answer.reasoning:
        sc -= 1
    if "850" not in answer.reasoning:
        sc -= 1
    return sc

# making a variable that stores the results version by version according to the keys in dict "prompts"
result = {version: [] for version in prompts.keys()}
scoring = {version: 0 for version in prompts.keys()} #var to store the score

#make the log header
print("prompt               ||              Score (out of 10)")

#making a loop, where prompts.items() in dict means to define the key and value by names so u can call them each
for ver, output in prompts.items():

    for idx, email in enumerate(test_emails):       #since test_emails was a list we call it one by one according to index
        formatted_prompt = output.format(email=email)   #fills each value with the current email in list
        response = model(formatted_prompt)      #inside loop to call the response on each mail
        count = score(response)
        scoring[ver] += count
        
        result[ver].append(response)    #didn't add the emails directly cuz we want name and response only
        time.sleep(0.5)                     # prevents APIs rate limiting (ex. only a request per sec), so pause for 5s
    print(f"{ver}               || {scoring[ver]} / 100") #out of 100 bcz 10 emails, each for 10 points

    #now for output
for ver, outputs in result.items():
    print(f"\n======== {ver} ========")
    for i, clean_text in enumerate(outputs):
        print(f"email {i+1}:\n{clean_text}\n" + "-"*40)
