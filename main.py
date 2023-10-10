import os
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from sentiment import analyze

app = FastAPI()

SECRET = os.getenv("SECRET")


@app.get("/")
def read_root():
    return PlainTextResponse(
        "sentiment analysis api\nmodel: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest\n\n~ nathaniel fernandes"
    )


@app.get("/sentiment/{text}")
def sentiment(req: Request, text: str):
    # check header for api key
    if req.headers.get("SECRET") != SECRET:
        return PlainTextResponse("unauthorized", status_code=401)

    print(f"Analyzing sentiment of `{text}`... 🤔")
    return analyze(text)
