# Copyright © 2026 Pathway

"""TwelveLabs components for the Pathway Live Data Framework.

This module provides two opt-in building blocks that let Pathway pipelines work
directly with video:

* :class:`TwelveLabsVideoParser` - a Pathway parser (``pw.UDF``) that turns raw
  video bytes into text using TwelveLabs' `Pegasus
  <https://docs.twelvelabs.io/docs/concepts/models/pegasus>`_ video-understanding
  model. The resulting text can be chunked, embedded and indexed by any of the
  standard Pathway RAG components, exactly like the output of the built-in PDF
  parsers.
* :class:`MarengoEmbedder` - a Pathway embedder (``BaseEmbedder``) backed by
  TwelveLabs' `Marengo <https://docs.twelvelabs.io/docs/concepts/models/marengo>`_
  multimodal embedding model. It returns 512-dimensional vectors that live in the
  same embedding space for text, image, audio and video, which makes it a natural
  choice when indexing the text produced by ``TwelveLabsVideoParser``.

Both components require the official ``twelvelabs`` Python SDK (``>=1.2.8``) and a
TwelveLabs API key. The key is read from the ``TWELVELABS_API_KEY`` environment
variable unless it is passed explicitly to the constructor.
"""

import asyncio
import logging
import os
import time

import numpy as np
import pathway as pw
from pathway import udfs
from pathway.xpacks.llm.embedders import BaseEmbedder

DEFAULT_PEGASUS_MODEL = "pegasus1.5"
DEFAULT_MARENGO_MODEL = "marengo3.0"
DEFAULT_PROMPT = (
    "Describe this video in detail. Summarize what happens, who and what appears, "
    "the setting, any spoken or on-screen text, and the overall topic. "
    "Write the description so it can be used to answer questions about the video."
)

logger = logging.getLogger(__name__)


def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("TWELVELABS_API_KEY")
    if not key:
        raise ValueError(
            "TwelveLabs API key is missing. Pass `api_key=...` or set the "
            "`TWELVELABS_API_KEY` environment variable."
        )
    return key


def _build_client(api_key: str | None):
    try:
        from twelvelabs import TwelveLabs
    except ImportError as e:
        raise ImportError(
            "The `twelvelabs` package is required to use the TwelveLabs components. "
            "Install it with `pip install twelvelabs>=1.2.8`."
        ) from e
    return TwelveLabs(api_key=_resolve_api_key(api_key))


def _build_async_client(api_key: str | None):
    try:
        from twelvelabs import AsyncTwelveLabs
    except ImportError as e:
        raise ImportError(
            "The `twelvelabs` package is required to use the TwelveLabs components. "
            "Install it with `pip install twelvelabs>=1.2.8`."
        ) from e
    return AsyncTwelveLabs(api_key=_resolve_api_key(api_key))


class TwelveLabsVideoParser(pw.UDF):
    """Parse videos into text using the TwelveLabs Pegasus model.

    The parser uploads the incoming video bytes to TwelveLabs as an asset, waits
    for the asset to be ready, and then asks Pegasus to produce a textual
    description of the video using ``prompt``. The returned text is suitable for
    chunking, embedding and indexing by the standard Pathway RAG components.

    By default the uploaded asset is deleted once the analysis finishes (even if
    the analysis fails), so repeated runs do not flood the TwelveLabs asset list.
    Set ``delete_assets=False`` to keep the assets around for reuse or
    inspection; in that case the emitted ``twelvelabs_asset_id`` metadata refers
    to a live, retrievable asset.

    Args:
        prompt: Instruction sent to Pegasus describing what to extract from the
            video. Defaults to a generic, RAG-oriented description prompt.
        model: Pegasus model name. Defaults to ``"pegasus1.5"``.
        api_key: TwelveLabs API key. If ``None``, the SDK reads it from the
            ``TWELVELABS_API_KEY`` environment variable.
        max_tokens: Maximum number of tokens Pegasus may generate. Defaults to 2048.
        temperature: Sampling temperature for Pegasus. Defaults to ``None`` (SDK default).
        asset_poll_interval: Seconds between asset-readiness checks. Defaults to 5.
        asset_timeout: Maximum number of seconds to wait for an uploaded asset to
            become ready before raising. Defaults to 600.
        delete_assets: If ``True`` (the default), the uploaded asset is deleted
            after the analysis completes, so repeated runs do not accumulate
            assets in your TwelveLabs account. When ``True``, the emitted
            ``twelvelabs_asset_id`` metadata is omitted because the asset no
            longer exists. Set to ``False`` to keep assets (e.g. for reuse or
            debugging), in which case the id is included in the metadata.
        cache_strategy: Pathway caching strategy. To enable caching, pass a valid
            :py:class:`~pathway.udfs.CacheStrategy`. Defaults to ``None``.

    Example:

    >>> import pathway as pw  # doctest: +SKIP
    >>> from pathway_twelvelabs import TwelveLabsVideoParser  # doctest: +SKIP
    >>> parser = TwelveLabsVideoParser()  # doctest: +SKIP
    """

    def __init__(
        self,
        prompt: str = DEFAULT_PROMPT,
        model: str = DEFAULT_PEGASUS_MODEL,
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float | None = None,
        asset_poll_interval: float = 5.0,
        asset_timeout: float = 600.0,
        delete_assets: bool = True,
        cache_strategy: udfs.CacheStrategy | None = None,
    ):
        super().__init__(cache_strategy=cache_strategy)
        self.prompt = prompt
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.asset_poll_interval = asset_poll_interval
        self.asset_timeout = asset_timeout
        self.delete_assets = delete_assets
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _build_client(self._api_key)
        return self._client

    def _upload_asset(self, contents: bytes) -> str:
        """Upload video bytes and return the asset id once it is ready."""
        asset = self.client.assets.create(
            method="direct", file=("video.mp4", contents), filename="video.mp4"
        )
        deadline = time.monotonic() + self.asset_timeout
        while asset.status not in ("ready", "failed"):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"TwelveLabs asset {asset.id} was not ready after "
                    f"{self.asset_timeout}s (last status: {asset.status})."
                )
            time.sleep(self.asset_poll_interval)
            asset = self.client.assets.retrieve(asset.id)
        if asset.status == "failed":
            raise RuntimeError(f"TwelveLabs asset {asset.id} failed to process.")
        return asset.id

    def __wrapped__(self, contents: bytes, **kwargs) -> list[tuple[str, dict]]:
        from twelvelabs.types.video_context import VideoContext_AssetId

        asset_id = self._upload_asset(contents)
        try:
            logger.info("Analyzing TwelveLabs asset %s with Pegasus...", asset_id)
            analyze_kwargs: dict = dict(
                model_name=self.model,
                video=VideoContext_AssetId(asset_id=asset_id),
                prompt=self.prompt,
                max_tokens=self.max_tokens,
            )
            if self.temperature is not None:
                analyze_kwargs["temperature"] = self.temperature
            response = self.client.analyze(**analyze_kwargs)
            text = response.data or ""
        finally:
            if self.delete_assets:
                # Remove the per-run asset so repeated runs do not flood the
                # TwelveLabs asset list. Best-effort: a cleanup failure must not
                # mask the analysis result (or an analysis error above).
                try:
                    self.client.assets.delete(asset_id)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to delete TwelveLabs asset %s.", asset_id)
        # When the asset has been deleted the id no longer resolves, so only
        # surface it in the metadata when the asset is kept around.
        metadata = {} if self.delete_assets else {"twelvelabs_asset_id": asset_id}
        return [(text, metadata)]

    def __call__(self, contents: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        """Parse the video document.

        Args:
            contents: Column with the raw bytes of each video.

        Returns:
            A column with a list of ``(text, metadata)`` pairs for each video.
            When ``delete_assets=False`` the metadata records the TwelveLabs
            ``twelvelabs_asset_id`` used for the analysis; with the default
            ``delete_assets=True`` the asset is removed afterwards and the id is
            omitted (it would no longer resolve).
        """
        return super().__call__(contents, **kwargs)


class MarengoEmbedder(BaseEmbedder):
    """Embed text using the TwelveLabs Marengo multimodal embedding model.

    Marengo returns 512-dimensional embeddings in a shared multimodal space, so the
    text it produces is directly comparable with image, audio and video embeddings
    from the same model. This makes it a natural retriever embedder for pipelines
    that index video with :class:`TwelveLabsVideoParser`.

    Args:
        model: Marengo model name. Defaults to ``"marengo3.0"``.
        api_key: TwelveLabs API key. If ``None``, the SDK reads it from the
            ``TWELVELABS_API_KEY`` environment variable.
        capacity: Maximum number of concurrent operations. Defaults to ``None``
            (no specific limit).
        retry_strategy: Strategy for handling retries. Defaults to
            :py:class:`~pathway.udfs.ExponentialBackoffRetryStrategy`.
        cache_strategy: Pathway caching strategy. Defaults to ``None``.

    Example:

    >>> import pathway as pw  # doctest: +SKIP
    >>> from pathway_twelvelabs import MarengoEmbedder  # doctest: +SKIP
    >>> embedder = MarengoEmbedder()  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MARENGO_MODEL,
        api_key: str | None = None,
        capacity: int | None = None,
        retry_strategy: (
            udfs.AsyncRetryStrategy | None
        ) = pw.udfs.ExponentialBackoffRetryStrategy(),
        cache_strategy: udfs.CacheStrategy | None = None,
    ):
        executor = udfs.async_executor(capacity=capacity, retry_strategy=retry_strategy)
        # Marengo embeds one text per request, so keep batches at size 1.
        super().__init__(
            executor=executor, cache_strategy=cache_strategy, max_batch_size=1
        )
        self.model = model
        self._api_key = api_key
        self._client = None
        self._aclient = None

    @property
    def client(self):
        if self._client is None:
            self._client = _build_client(self._api_key)
        return self._client

    @property
    def aclient(self):
        if self._aclient is None:
            self._aclient = _build_async_client(self._api_key)
        return self._aclient

    def get_embedding_dimension(self, **kwargs) -> int:
        """Return the embedding dimension (512 for Marengo).

        This is a one-time, setup-time probe: Pathway calls it once while
        building the index, not on the per-document hot path. The single
        synchronous request issued here is therefore intentional and acceptable
        (the actual embedding hot path runs asynchronously via ``__wrapped__``).

        The base implementation probes ``__wrapped__`` with a single string and
        takes ``len`` of the result; since this embedder always returns a list of
        vectors, probe with a one-element list and measure the first vector
        instead (mirroring Pathway's ``SentenceTransformerEmbedder``).
        """
        return len(self._embed_one("."))

    def _embed_one(self, text: str) -> np.ndarray:
        """Synchronous single-text embed, used only for the setup-time probe."""
        response = self.client.embed.create(model_name=self.model, text=text)
        vector = response.text_embedding.segments[0].float_
        return np.array(vector, dtype=np.float32)

    async def _aembed_one(self, text: str) -> np.ndarray:
        resp = await self.aclient.embed.create(model_name=self.model, text=text)
        vector = resp.text_embedding.segments[0].float_
        return np.array(vector, dtype=np.float32)

    async def __wrapped__(self, inputs: list[str], **kwargs) -> list[np.ndarray]:
        """Embed the given texts with Marengo.

        Marengo embeds one text per request, so the requests are issued
        concurrently on the async TwelveLabs client (``AsyncTwelveLabs``) rather
        than serially, keeping the embedding hot path non-blocking.

        Args:
            inputs: the strings to embed.

        Returns:
            A list of 512-dimensional ``numpy`` arrays, one per input string.
        """
        return list(await asyncio.gather(*[self._aembed_one(t) for t in inputs]))
