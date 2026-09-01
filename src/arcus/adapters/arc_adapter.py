import os
from enum import Enum

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletion


class ArcModel(str, Enum):
    """The four models this build knows how to route to and has cost/
    reward data for. ARC maintains its own catalog independently of this
    project and can add, remove, or rename entries at any time, so
    treat this as a snapshot rather than a guarantee, see
    routing/model_catalog.py for how the router cross-checks it against
    what ARC is actually serving.
    """

    GPT_OSS_120B = "gpt-oss-120b"
    GLM_5_3 = "GLM-5.3"
    KIMI_K3 = "Kimi-K3"
    DEEPSEEK_V4_FLASH = "DeepSeek-V4-Flash"


class ArcAdapter:
    """Thin wrapper around the openai SDK pointed at ARC's endpoint.

    ARC's API is OpenAI-compatible, so we don't need a custom HTTP client,
    just the right base_url and a key. Everything else (routing, caching,
    the quality gate) should go through this instead of importing openai
    directly, so ARC's endpoint details only live in one place.
    """

    BASE_URL = "https://llm-api.arc.vt.edu/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        api_key = api_key or os.environ.get("ARC_API_KEY")
        if not api_key:
            raise ValueError(
                "no ARC API key found. Pass api_key=<your-key> "
                "Get one from llm.arc.vt.edu under "
                "User profile > Settings > Account > API keys."
            )

        # timeout + max_retries gives us the "be a good citizen on shared
        # infra" behavior the spec asks for, no need to hand-roll a rate
        # limiter on top of what the SDK already does.
        self._client = OpenAI(
            base_url=self.BASE_URL,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def chat(
        self,
        model: ArcModel | str,
        messages: list[dict],
        **kwargs,
    ) -> ChatCompletion:
        """Send a chat completion request to ARC for the given model.

        Returns the raw ChatCompletion, not just the text, because callers
        further down the pipeline (the quality gate especially) need
        finish_reason and usage info, not just the message content.
        """
        return self._client.chat.completions.create(
            model=model.value if isinstance(model, ArcModel) else model,
            messages=messages,
            **kwargs,
        )

    def list_models(self) -> list[str]:
        """Model ids ARC is actually serving right now, straight from its
        catalog endpoint rather than anything hardcoded on our side.
        """
        return [entry.id for entry in self._client.models.list()]

    def upload_file(self, path: str) -> str:
        """Uploads a file to ARC's RAG endpoint and returns its file id,
        for referencing in a later chat request's `files` parameter.

        Not available through the openai SDK's own files API, ARC's
        upload endpoint is its own multipart POST rather than an
        OpenAI-compatible one, so this goes over httpx directly.
        """
        with open(path, "rb") as f:
            response = httpx.post(
                f"{self.BASE_URL}/files/",
                headers={"Authorization": f"Bearer {self._client.api_key}", "Accept": "application/json"},
                files={"file": f},
                timeout=60.0,
            )
        response.raise_for_status()
        return response.json()["id"]

    def delete_file(self, file_id: str) -> None:
        """Removes a previously uploaded file from the user's ARC account.
        Best-effort cleanup, callers that don't want a failure here to be
        fatal should catch httpx.HTTPError around this themselves.
        """
        response = httpx.delete(
            f"{self.BASE_URL}/files/{file_id}",
            headers={"Authorization": f"Bearer {self._client.api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()