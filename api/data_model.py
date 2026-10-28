"""Request and response contracts shared by the three inference endpoints."""

from typing import Annotated

from pydantic import BaseModel, EmailStr, Field, HttpUrl, StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10000)]


class NLPDataInput(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=32)
    user_id: EmailStr


class ImageDataInput(BaseModel):
    urls: list[HttpUrl] = Field(min_length=1, max_length=8)
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
