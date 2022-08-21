from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dataclasses import dataclass
from pydantic import BaseModel
from analyzer import Analyzer


class PredictRequest(BaseModel):
    sentence: str


@dataclass
class PredictResponse:
    character: str


character_dict = {
    0: "香風智乃",
    1: "保登心愛",
    2: "天々座理世",
    3: "桐間紗路",
    4: "宇治松千夜",
    5: "条河麻耶",
    6: "奈津恵",
}
analyzer = Analyzer("model", character_dict)
app = FastAPI(
    title="Sora Serifu Predict",
    description="台詞からキャラクターを推定するサーバー",
    version="0.1.0",
)


@app.get("/health_check")
async def health_check():
    """生存確認用エンドポイント"""
    return {"message": "Server is working"}


@app.post("/predict", response_model=PredictResponse)
async def predict_difficulty(req: PredictRequest) -> PredictResponse:
    """テキストから応答を推定するエンドポイント"""
    resp = analyzer.get_character_from_sentence(req.sentence)
    return PredictResponse(character=resp)


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, debug=True)
