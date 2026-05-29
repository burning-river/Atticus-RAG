import os

from huggingface_hub import AsyncInferenceClient
import asyncio
from threading import Thread
# from transformers import TextIteratorStreamer

HF_TOKEN = os.getenv("HF_API_TOKEN")

client = AsyncInferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=HF_TOKEN
)
# pipe = pipeline(
#     "text-generation",
#     model=chat_model,
#     device_map="auto",
#     max_new_tokens=50,
#     do_sample=False,
# )


def build_prompt(query: str, chunks: list[str]) -> list[dict]:
    if not chunks:
        return [{"role": "user", "content": "No relevant context was found."}]

    context = chunks[0]
    return [
        {
            "role": "system",
            "content": """You are an assistant that extracts specific facts from legal text.
    Answer the question based only on the provided context.
    If the information is not explicitly or implicitly mentioned, return 'Information not available'.
    Be concise.
        """,
        },
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
    ]


async def stream_response(messages: list[dict]):
    try:
        # Request a streaming completion from the API
        response_stream = await client.chat_completion(
            messages=messages,
            max_tokens=50,      # Equivalent to max_new_tokens
            temperature=0.0,    # Equivalent to do_sample=False (deterministic)
            stream=True
        )

        # Iterate over chunks as they arrive from the network
        async for chunk in response_stream:
            new_text = chunk.choices[0].delta.content
            if new_text:
                yield new_text
                # Optional: minimal sleep to match your original pacing loop
                await asyncio.sleep(0.01)

    except Exception as e:
        yield f"Error: {str(e)}"