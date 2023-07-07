from fastapi import FastAPI, Request
import uvicorn, numpy as np
import random, cv2, os
from io import BytesIO
import base64, time
from PIL import Image

app = FastAPI()

start_str = 'data:image/png;base64,'
lens = len(start_str)

def stringToRGB(base64_string):
    if base64_string[:lens] == start_str:
        base64_string= base64_string[lens:]
    img_data = base64.b64decode(str(base64_string))
    image = Image.open(BytesIO(img_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


count = [0]
current_folder = os.path.dirname(os.path.realpath(__file__))
@app.post("/movie_controller")
async def detect(request: Request):
    json_file: bytes = await request.json()
    imageSrc = json_file['imageSrc']
    image_np = stringToRGB(imageSrc)
    img_path = os.path.join(current_folder, "img.png")
    if os.path.exists(img_path):
        os.remove(img_path)
    cv2.imwrite(img_path, image_np)
    i = 101
    count[0] += 1
    if (count[0] % 10 == 0):
        i = 5
    # count[0] = 100
    # print(f"SEND DATA {'mute'}")
    return {"content": f"100"}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)