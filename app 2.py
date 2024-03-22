from flask import Flask, render_template, request, jsonify 
from datetime import datetime
import json
import requests
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

app = Flask(__name__)

LEAVE_TYPES = {
    0: "Sick Leave",
    1: "Emergency Leave",
    2: "Casual Leave",
    3: "Earned Leave",
    4: "Vacation Leave",
    5: "Half Day"
}

DB_FAISS_PATH = 'vectorstore/db_faiss'
SAVE_RECORD_ENDPOINT = "http://192.168.1.48:7004/api/Bot/SaveRecord"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Loading the model and embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': 'cpu'})
llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_type="llama",
    max_new_tokens=512,
    temperature=0.5
)

# Loading the vector store
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Prompt template for QA retrieval
qa_prompt = set_custom_prompt()

# Retrieval QA Chain
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type='stuff',
                                 retriever=db.as_retriever(search_kwargs={'k': 2}),
                                 return_source_documents=True,
                                 chain_type_kwargs={'prompt': qa_prompt}
                                 )


def fetch_leave_types():
    try:
        response = requests.get("http://192.168.1.48:7004/api/Bot/GetList")
        if response.status_code == 200:
            leave_types = response.json()
            global LEAVE_TYPES
            LEAVE_TYPES = {leave['leaveId']: leave['leaveDesc'] for leave in leave_types}  # Assuming the API returns JSON data containing leave types with 'leaveId' and 'leaveDesc' keys
        else:
            print("Failed to fetch leave types.")
    except Exception as e:
        print(f"Error occurred while fetching leave types: {e}")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def get_response():
    user_input = request.form["msg"]
    if user_input == "1":  # If user chooses option 1 (apply for leave)
        fetch_leave_types()
        response = chat(user_input)
        return response
    else:
         response = chat(user_input)
         return response # Return response as JSON

# Output function
def final_result(query):
    response = qa({'query': query})
    
    # Extract the text content from the response
    text_content = response['result']
    return text_content

def chat(user_input):
    global conversation_state
    
    if "conversation_state" not in globals():
        conversation_state = {}

    if user_input.lower() in ['hello', 'hi', 'hey']:
        return "Welcome! How can I assist you?\n1. Apply for Leave\n2. Privacy Policy"
    elif conversation_state.get("step") == "apply_leave":
        return handle_apply_leave(user_input)
    elif user_input.lower() == '1':
        conversation_state["step"] = "apply_leave"
        return "Leave application initiated. Please provide the required information.\nDate from: (MM/DD/YYYY)"
    else:
        # If not a special command, interact with LLM
        return final_result(user_input)

def handle_apply_leave(user_input):
    global conversation_state

    if "conversation_state" not in globals():
        conversation_state = {}

    if "date_from" not in conversation_state:
        try:
            date_from = datetime.strptime(user_input, "%m/%d/%Y")
            conversation_state["date_from"] = date_from
            return "Date to: (MM/DD/YYYY)"
        except ValueError:
            return "Invalid date format. Please enter date in MM/DD/YYYY format."

    elif "date_to" not in conversation_state:
        try:
            date_to = datetime.strptime(user_input, "%m/%d/%Y")
            if date_to < conversation_state["date_from"]:
                return "Date to should not be smaller than Date from. Please enter again."
            else:
                conversation_state["date_to"] = date_to
                # Show leave IDs and descriptions to the user
                leave_type_options = "\n".join(f"{leave_id} - {LEAVE_TYPES[leave_id]}" for leave_id in LEAVE_TYPES.keys())
                return f"Select type of leave:\n{leave_type_options}"
        except ValueError:
            return "Invalid date format. Please enter date in MM/DD/YYYY format."

    elif "leave_type" not in conversation_state:
        try:
            leave_type_input = user_input.split(' - ')
            leave_type = int(leave_type_input[0])
            if leave_type not in LEAVE_TYPES:
                return "Invalid leave type. Please select from the given options."
            else:
                conversation_state["leave_type"] = leave_type
                return "Reason:"
        except ValueError:
            return "Invalid input. Please enter a valid leave type in the format 'leaveID - leaveDescription'."

    elif "reason" not in conversation_state:
        conversation_state["reason"] = user_input
        response = f"Please confirm the details:\nDate From: {conversation_state['date_from'].strftime('%m/%d/%Y')}\nDate To: {conversation_state['date_to'].strftime('%m/%d/%Y')}\nType of Leave: {conversation_state['leave_type']} - {LEAVE_TYPES[conversation_state['leave_type']]}\nReason: {conversation_state['reason']}\n\nType 'confirm' to proceed or 'cancel' to abort."
        return response

    elif user_input.lower() == "confirm":
        leave_data = {
            "DateFrom": conversation_state["date_from"].strftime('%m-%d-%Y'),
            "DateTo": conversation_state["date_to"].strftime('%m-%d-%Y'),
            "Type": conversation_state["leave_type"],
            "Remarks": conversation_state["reason"]
        }
        response = save_leave_data(leave_data)
        # Here you can process the leave application, for now, let's just reset the conversation state
        conversation_state.clear()
        return response

    elif user_input.lower() == "cancel":
        conversation_state.clear()
        return "Leave application cancelled."

    else:
        return "Sorry, I didn't understand that. Could you please repeat or choose from the options?"
    
def save_leave_data(leave_data):
    try:
        payload_json = json.dumps(leave_data)  # Convert the dictionary to a JSON string
        print("Payload sent to backend:", payload_json)  # Print the payload
        response = requests.post(SAVE_RECORD_ENDPOINT, json=leave_data)
        if response.status_code == 200:
            return "Leave data saved successfully."
        else:
            return f"Failed to save leave data. Status code: {response.status_code}"
    except Exception as e:
        return f"Error occurred while saving leave data: {e}"
    
    
if __name__ == '__main__':
    app.run(debug=True)
