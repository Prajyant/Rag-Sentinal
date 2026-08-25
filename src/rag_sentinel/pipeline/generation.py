"""LLM generation layer — wraps Ollama for local inference."""

from __future__ import annotations

from loguru import logger


class LLMGenerator:
    """Sends prompts to a locally-running Ollama model.

    Parameters
    ----------
    model:
        Ollama model name, e.g. "llama3.2", "mistral", "phi3".
    host:
        Ollama API base URL.
    temperature:
        Sampling temperature.  0.0 = deterministic for eval.
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer the user's question using only "
        "the provided context. If the context does not contain the answer, "
        "say so clearly. Do not make up information."
    )

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.host = host
        self.temperature = temperature

    def generate(self, query: str, context_chunks: list[str]) -> str:
        """Generate an answer grounded in the provided context chunks."""
        try:
            import ollama
        except ImportError as exc:
            raise ImportError("ollama is required: pip install ollama") from exc

        context = "\n\n---\n\n".join(context_chunks)
        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        client = ollama.Client(host=self.host)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": self.temperature},
        )
        answer = response["message"]["content"]
        logger.debug(f"Generated answer ({len(answer)} chars)")
        return answer

    def __call__(self, prompt: str) -> str:
        """Simple callable interface for the consistency checker."""
        try:
            import ollama
            client = ollama.Client(host=self.host)
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            return response["message"]["content"]
        except Exception as exc:
            logger.warning(f"LLM call failed: {exc}")
            return ""
