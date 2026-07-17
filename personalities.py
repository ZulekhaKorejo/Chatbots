import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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

behaviors = SystemMessage(content=(mode))

memory=[behaviors]
print("Enter 0 to exit the convo")
while True:
    myprompt=input("YOU: ")
    if myprompt =="0":
        break
    questions = HumanMessage(content=(myprompt))
    memory.append(questions)
    response=model1.invoke(memory)
    answer = AIMessage(content=response.content)
    memory.append(answer)
    print(answer.content)