import os
import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

# Reuse the existing inference utilities
from gspeech import predict, sampling_rate, CONFIDENCE_THRESHOLD, processor

app = FastAPI(title="Voice Safety Model API", version="1.0.0")

API_KEY_ENV = "VOICE_API_KEY"


def verify_api_key(x_api_key: Optional[str]) -> None:
	expected = os.getenv(API_KEY_ENV)
	if not expected:
		raise HTTPException(status_code=500, detail=f"Server is not configured. Set {API_KEY_ENV} env var.")
	if x_api_key != expected:
		raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/detect")
async def detect(audio: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
	verify_api_key(x_api_key)
	# Save to a temporary in-memory buffer then to disk path expected by predict()
	data = await audio.read()
	if not data:
		raise HTTPException(status_code=400, detail="Empty file")
	# Write to a temporary file path
	tmp_path = f"/tmp/{audio.filename}"
	with open(tmp_path, "wb") as f:
		f.write(data)

	results = predict(tmp_path, sampling_rate)
	top = max(results, key=lambda x: x["Score"]) if results else None
	unsafe = bool(top and (top["Vocalization"].lower() == "screaming" and top["Score"] > CONFIDENCE_THRESHOLD))
	return JSONResponse({
		"unsafe": unsafe,
		"top": top,
		"threshold": CONFIDENCE_THRESHOLD
	})


@app.get("/health")
async def health():
	return {"status": "ok"}


if __name__ == "__main__":
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)

