import requests
import os

url = "http://localhost:8000/predict/"

file_path = "path/to/fname.h5"
os.makedirs("./preds", exist_ok=True)

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "application/octet-stream")}
    resp = requests.post(url, files=files)

if resp.status_code == 200:
    try:
        fname = resp.headers['X-Filename']
    except KeyError:
        raise KeyError("Response is missing 'X-Filename' header")
    
    output_path = os.path.join("./preds", fname)
    with open(output_path, "wb") as out:
        out.write(resp.content)
    print(f"Prediction saved as {output_path}")
else:
    print("Error:", resp.status_code, resp.text)