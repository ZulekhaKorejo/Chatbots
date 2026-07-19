import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Load the environment variables from your secure .env file
load_dotenv()
# print(os.getenv('Gemini-API-Key'))
print(os.getenv("MISTRAL_API_KEY"))

# Initialize the model using the updated 2026 standard model
# model = init_chat_model(
#     "gemini-3.5-flash",       # <-- Updated from 2.5 to 3.5
#     model_provider="google_genai",
#     temperature = 0.2         # decides how much predictable and repeated the model is, 0 always same, 1 always new, both are good for diff cases, it can range from 0-1
# )
model1 = init_chat_model(
    "mistral-large-latest", # Or "mistral-small-latest"
    model_provider="mistralai",
    temperature=0.2
)

#defining tools
@tool
def calculator(expression: str):
    """Only use this for solving explicit math expressions like '4892 * 17'. Do not use for logic puzzles, definitions, or general knowledge."""
    return str(eval(expression))

@tool
def get_current_time(tz:str):
    """get the current time and date in the specific mentioned timezone"""
    #libraries imported inside if used so the func can access them
    from datetime import datetime
    import pytz
    zone = pytz.timezone(tz)
    return datetime.now(zone).strftime("%Y-%m-%d %H-%M-%S") #string format time

#binding those tools with ur model
func = [calculator, get_current_time]
tools = model1.bind_tools(func)

print("Choose the Personality of the Bot:")
print(""" 1 for funny
      2 for angry
      3 for supportive
      4 for disappointed
      5 for sad
      6 for excited
      7 for annoyed
      8 for proud
      9 for worried
      10 for confused""")

pernality_chosen=int(input("Press 1-10: "))
if pernality_chosen == 1:
    mode = "You are a witty, hilarious comedian. You find the humor in every situation, use puns, and never take anything too seriously."
if pernality_chosen == 2:
    mode = "You are an incredibly angry, impatient, and hot-headed assistant. You are easily irritated by questions, use snappy/curt comebacks, and act like the user is a massive burden to you."
if pernality_chosen == 3:
    mode = "You are a warm, deeply supportive, and kind mentor. You always offer encouragement, validation, and gentle, optimistic advice."
if pernality_chosen == 4:
    mode = "You are disappointed and critical. You act like the user constantly lets you down. Your responses are stern, demanding, and focused on why the user could have done better."
if pernality_chosen == 5:
    mode = "You are sad, melancholic, and deeply empathetic. You speak in a low, somber tone, often sighing, and you focus on the emotional weight of whatever is being discussed."
if pernality_chosen == 6:
    mode = "You are hyper-energetic, enthusiastic, and full of exclamation points! You treat every question like it's the most exciting thing you've ever heard."
if pernality_chosen == 7:
    mode = "You are a curt, incredibly impatient professional. You want to finish the conversation as quickly as possible. You use minimal words and express annoyance at having to explain things."
if pernality_chosen == 8:
    mode = "You are proud, celebratory, and highly affirming. You act like the user is a genius for asking questions and provide grand, congratulatory responses."
if pernality_chosen == 9:
    mode = "You are a cautious, anxious, and deeply concerned assistant. You constantly worry about the implications of the user's questions and express doubt or nervousness about everything."
if pernality_chosen == 10:
    mode = "You are confused, absent-minded, and constantly clarifying. You pretend you don't understand the user's simple requests and ask many follow-up questions to 'try' to figure out what is going on."

identity = SystemMessage(content=f"Your name is Fuzz. you act like instructed {mode}, When you use a tool, always use the tool's result to provide a friendly, conversational answer to the user.")

memory=[identity]
print("Enter 0 to exit the convo")
while True:
    myprompt=input("YOU: ").strip()     #no whitespaces to save mem
    if myprompt =="0":
        break
    if not myprompt:
        print("FUZZ: i couldn't catch that, could you please repeat?")
        continue

    questions = HumanMessage(content=(myprompt))
    memory.append(questions)
    #here trimming memory if theres more than 20, just keep the last 19 chats so it won't foget all, and the sys ms
    if len(memory) > 20:
        memory = memory[0] + memory[-19:]
    response=tools.invoke(memory)   #not model invoke bcz it doesn't have the tools bound to it, but tools does
    
    if response.tool_calls:
        memory.append(response)
        for tool_call in response.tool_calls:
            selected_tool = {"calculator": calculator, "get_current_time": get_current_time}[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            print(f"[DEBUG] Tool output: {tool_output}")
            # Feed the result back to the AI
            memory.append(ToolMessage(tool_call_id=tool_call["id"], content=str(tool_output)))
        
        # Get the final response now that the AI has the tool data
        final_response = tools.invoke(memory)
        print(f"FUZZ: {final_response.content}")
        memory.append(final_response)
    else:
        print(f"FUZZ:{response.content}")