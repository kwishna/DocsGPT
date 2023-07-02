import json
import requests
API_URL = "https://api-inference.huggingface.co/models/gpt2"
API_TOKEN = None
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))
data = query("Can you please let us know more details about your ")
# data = query({"inputs":"Can you please let us know more details about your "})