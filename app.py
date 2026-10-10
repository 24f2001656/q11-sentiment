
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

# Allow requests from the assignment website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Define the expected JSON request
class SentimentRequest(BaseModel):
    sentences: list[str]


def analyze_sentiment(sentence: str) -> str:
    score = analyzer.polarity_scores(sentence)["compound"]

    if score >= 0.05:
        return "happy"
    elif score <= -0.05:
        return "sad"
    else:
        return "neutral"


@app.post("/sentiment")
async def sentiment(request: SentimentRequest):
    results = []

    for sentence in request.sentences:
        results.append({
            "sentence": sentence,
            "sentiment": analyze_sentiment(sentence)
        })

    return {"results": results}


@app.get("/")
async def root():
    return {"message": "Sentiment API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
