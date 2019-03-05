from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
from fastai.vision import *
from segal_utils import *
import base64

model_file_url = 'https://www.dropbox.com/s/grsnoaj10mojots/amit_generate.pkl?raw=1'
model_file_name = 'amit_generate.pkl'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


async def setup_learner(model_file_url, model_file_name):
    await download_file(model_file_url, path / 'models' / f'{model_file_name}')
    learn = load_learner(path / 'models', model_file_name)
    return learn


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner(model_file_url, model_file_name))]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


PREDICTION_FILE_SRC = path / 'static' / 'predictions.txt'


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    text = data["text"]
    try:
        number =  data["number"]
    except:
        number =  1
    # bytes = base64.b64decode(img_bytes)
    return predict_from_bytes(str(text), int(number))


def predict_from_bytes(text, number):
    # img = open_image(BytesIO(bytes))
    predicter = NextWord(learn, text)
    predicter.generate(len(text.split(" ")) + number)
    result = predicter.sentence
    # predictions = sorted(zip(classes, map(float, losses)), key=lambda p: p[1], reverse=True)
    result_html1 = path / 'static' / 'result1.html'
    result_html2 = path / 'static' / 'result2.html'
    result_html = str(result_html1.open().read() + str(result) + result_html2.open().read())
    return HTMLResponse(result_html)


@app.route("/")
def form(request):
    index_html = path / 'static' / 'index.html'
    return HTMLResponse(index_html.open().read())


if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)