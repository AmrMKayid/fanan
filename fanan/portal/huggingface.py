import os
import tempfile
import urllib
import urllib.parse
from dataclasses import dataclass

import fsspec
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

PYTORCH_MODEL = "pytorch_model.bin"
SAFE_TENSORS_MODEL = "model.safetensors"


@dataclass(frozen=True)
class RepoRef:
    model_name_or_path: str
    revision: str | None = None

    @staticmethod
    def from_string(name: str) -> "RepoRef":
        if "@" not in name:
            return RepoRef(name)
        model_name_or_path, revision = name.split("@")
        return RepoRef(model_name_or_path, revision)

    def __str__(self) -> str:
        if self.revision is None:
            return self.model_name_or_path
        return f"{self.model_name_or_path}@{self.revision}"

    def __repr__(self) -> str:
        return f"RemoteRev({self.model_name_or_path!r}, {self.revision!r})"


def _is_url_like(path: str) -> bool:
    return urllib.parse.urlparse(path).scheme != ""


def load_tokenizer(
    model_name_or_path: str,
    revision: str | None = None,
    local_cache_dir: str | None = None,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load a tokenizer from a model name or path, with optional revision and
    local cache directory."""
    if _is_url_like(model_name_or_path):
        if revision is not None:
            raise ValueError("revision is not supported for URLs")

        if local_cache_dir is None:
            local_cache_dir = tempfile.mkdtemp()

        fs, path = fsspec.core.url_to_fs(model_name_or_path)
        fs.get(path, local_cache_dir, recursive=True)
        base_path = os.path.basename(path)

        return AutoTokenizer.from_pretrained(
            os.path.join(local_cache_dir, base_path),
            trust_remote_code=trust_remote_code,
        )

    return AutoTokenizer.from_pretrained(model_name_or_path, revision=revision, trust_remote_code=trust_remote_code)
