"""
Provider de embeddings usando OpenAI API.

Demonstra como adicionar um provider com retry/backoff e rate limiting.
"""

from typing import Dict, List, Optional
import numpy as np
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import openai

from .base import EmbeddingProvider, register_provider


class OpenAIProvider(EmbeddingProvider):
    """
    Provider de embeddings usando OpenAI API.
    
    Configuracao exemplo:
    {
        "model_name": "text-embedding-3-small",
        "api_key": "sk-...",  # ou via env OPENAI_API_KEY
        "batch_size": 100,  # max recomendado pela API
        "max_retries": 3,
        "normalize": True
    }
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_name = config.get("model_name", "text-embedding-3-small")
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.batch_size = config.get("batch_size", 100)
        self.max_retries = config.get("max_retries", 3)
        self.normalize = config.get("normalize", True)
        
        if not self.api_key:
            raise ValueError("OpenAI API key nao fornecida (config ou OPENAI_API_KEY)")

        client_kw: Dict = {"api_key": self.api_key}
        base_url = config.get("base_url")
        if base_url:
            client_kw["base_url"] = str(base_url).strip()
        self.client = openai.OpenAI(**client_kw)
        self._cached_dim: Optional[int] = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed um batch com retry automatico."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        if not response.data:
            raise ValueError(
                f"No embedding data received for batch of {len(texts)} texts "
                f"(model={self.model_name}). Possible token limit exceeded."
            )

        embeddings = np.array([item.embedding for item in response.data])

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def _discover_dim(self) -> int:
        """Descobre a dimensao do embedding usando um texto trivial."""
        probe = self._embed_batch(["hello"])
        return probe.shape[1]

    def _embed_single_fallback(self, texts: List[str]) -> np.ndarray:
        """Fallback: embed one text at a time when batch fails.

        Descobre a dimensao do embedding antes de processar, garantindo que
        textos que excedem o context window recebam vetor-zero em vez de
        causar crash.
        """
        if self._cached_dim is None:
            self._cached_dim = self._discover_dim()
        dim = self._cached_dim

        results: List[np.ndarray] = []
        for text in texts:
            try:
                emb = self._embed_batch([text])
                results.append(emb[0])
            except Exception:
                print(
                    f"  [fallback] Text too long ({len(text)} chars), "
                    f"using zero vector (dim={dim})."
                )
                results.append(np.zeros(dim, dtype=np.float32))
        return np.vstack(results)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings usando OpenAI API com batching e retry.

        Se um batch inteiro falhar (ex: texto excede context window),
        tenta individualmente como fallback.

        Args:
            texts: Lista de strings

        Returns:
            Array numpy (n_texts, embedding_dim) normalizado
        """
        if not texts:
            return np.array([]).reshape(0, 0)

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                batch_embeddings = self._embed_batch(batch)
            except Exception:
                batch_embeddings = self._embed_single_fallback(batch)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)


# Registra automaticamente
register_provider("openai", OpenAIProvider)
