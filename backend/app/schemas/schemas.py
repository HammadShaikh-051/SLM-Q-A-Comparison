from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    model: str = Field(..., description="The name of the model to use (e.g., distilgpt2)")
    context: str = Field(..., description="The context text for the Q&A")
    question: str = Field(..., description="The question to ask based on the context")

class PredictResponse(BaseModel):
    model: str
    answer: str
    inference_time_seconds: float
