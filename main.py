import os
import time
import httpx
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

model_ready = False


async def wait_for_ollama(timeout: int = 120) -> bool:
    """Poll Ollama /api/tags until it responds, then pull the default model."""
    global model_ready
    deadline = time.time() + timeout
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            return False

        # Pull model
        try:
            await client.post(
                f"{OLLAMA_URL}/api/pull",
                json={"name": DEFAULT_MODEL, "stream": False},
                timeout=300.0,
            )
            model_ready = True
            return True
        except Exception:
            return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(wait_for_ollama())
    yield


app = FastAPI(
    title="Ollama Inference API",
    description="Lightweight LLM inference endpoint powered by Ollama on Varity",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    max_tokens: int = 256


class GenerateResponse(BaseModel):
    response: str
    model: str


@app.get("/")
async def root():
    return {
        "service": "Ollama Inference API",
        "version": "1.0.0",
        "status": "ready" if model_ready else "warming_up",
        "model": DEFAULT_MODEL,
        "ollama_url": OLLAMA_URL,
    }


@app.get("/health")
async def health():
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return {"status": "ready", "model": DEFAULT_MODEL}


@app.get("/models")
async def list_models():
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    async with httpx.AsyncClient() as client:
        # Lazy pull: ensure model is available
        try:
            await client.post(
                f"{OLLAMA_URL}/api/pull",
                json={"name": req.model, "stream": False},
                timeout=120.0,
            )
        except Exception:
            pass

        try:
            r = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": req.model,
                    "prompt": req.prompt,
                    "stream": False,
                    "options": {"num_predict": req.max_tokens},
                },
                timeout=120.0,
            )
            r.raise_for_status()
            data = r.json()
            return GenerateResponse(
                response=data.get("response", ""),
                model=req.model,
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Ollama error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Inference failed: {e}")
