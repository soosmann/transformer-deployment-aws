from pydantic import BaseModel, EmailStr

class NLPDataInput(BaseModel):
    texts: list[str]
    user_id: EmailStr

class ImageDataInput(BaseModel):
    urls: list[str]
    user_id: EmailStr

class NLPDataOutput(BaseModel):
    pred_model: str
    texts: list[str]
    labels: list[str]
    scores: list[float]
    prediction_time: int

class ImageDataOutput(BaseModel):
    pred_model: str
    urls: list[str]
    labels: list[str]
    scores: list[float]
    prediction_time: int