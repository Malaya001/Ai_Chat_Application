import requests

def get_llama3(prompt):
    ollama_models = "http://localhost:11434/api/generate"
    payload = {
        "model":"llama3",
        "prompt":prompt,
        "stream":False,
        "options":{
            "temperature":1
        }
    } 
    response = requests.post(ollama_models, json = payload)
    print(response)


    json_response = response.json()
    response_text = json_response.get("response", "No response field found")

    # chat_history = ChatHistory(user_input = prompt, modle_response = response_text)
    return response_text

def get_llama3_1(prompt):
    ollama_models = "http://localhost:11434/api/generate"
    payload = {
        "model":"llama3.1",
        "prompt":prompt,
        "stream":False,
        "options":{
            "temperature":1
        }
    } 
    response = requests.post(ollama_models, json = payload)
    print(response)


    json_response = response.json()
    response_text = json_response.get("response", "No response field found")

    # chat_history = ChatHistory(user_input = prompt, modle_response = response_text)
    return response_text


def get_dolphin_phi(prompt):
    ollama_models = "http://localhost:11434/api/generate"
    payload = {
        "model":"dolphin-phi",
        "prompt":prompt,
        "stream":False,
        "options":{
            "temperature":1
        }
    } 
    response = requests.post(ollama_models, json = payload)
    print(response)


    json_response = response.json()
    response_text = json_response.get("response", "No response field found")

    # chat_history = ChatHistory(user_input = prompt, modle_response = response_text)
    return response_text
