import time
from app.schemas.schemas import PredictRequest, PredictResponse
from app.models.model_manager import model_manager

def run_inference(request: PredictRequest) -> PredictResponse:
    model_id = request.model.lower()
    
    # Fetch from manager (will load lazily or return cached)
    model, tokenizer, device = model_manager.get_model(model_id)
    
    if model is None or tokenizer is None:
        raise RuntimeError(f"Model '{model_id}' is not loaded properly.")

    start_time = time.time()
    
    prompt = f"### Context:\n{request.context}\n\n### Question:\n{request.question}\n\n### Answer:\n"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    # Lazy import of torch to avoid breaking the fast startup requirement
    import torch
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id
        )

    # Remove input tokens and decode only generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    inference_time = round(time.time() - start_time, 2)
    
    return PredictResponse(
        model=request.model,
        answer=answer,
        inference_time_seconds=inference_time
    )
