"""Tokenize source text, encode it once and generate target tokens."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch import Tensor

from ..data.masks import cross_attention_mask, decoder_attention_mask, encoder_attention_mask
from ..data.tokenizer import BilingualBPETokenizer
from ..model.transformer import EncoderDecoderTransformer
from ..training.utils import select_device
from .beam_search import beam_search_decode
from .checkpoint_loader import load_translation_model
from .greedy import Hypothesis, decode_tokens, greedy_decode
from .sampling import sample_next_token


@dataclass(frozen=True)
class InferenceConfig:
    strategy: str = "greedy"
    max_new_tokens: int = 64
    beam_width: int = 4
    length_penalty: float = 0.6
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    device: str = "auto"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.strategy not in ("greedy", "beam", "sampling"):
            raise ValueError("strategy must be greedy, beam or sampling")
        for name in ("max_new_tokens", "beam_width"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.length_penalty) not in (float, int) or not math.isfinite(self.length_penalty) or self.length_penalty < 0:
            raise ValueError("length_penalty must be finite and non-negative")
        if type(self.temperature) not in (float, int) or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if type(self.top_p) not in (float, int) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be auto, cpu, cuda or mps")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError("seed must be a non-negative 63-bit integer")

    @classmethod
    def load(cls, path: Path) -> "InferenceConfig":
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or not isinstance(config.get("inference"), dict):
                raise ValueError("Expected an inference mapping")
            return cls(**config["inference"])
        except (TypeError, ValueError, yaml.YAMLError) as error:
            raise ValueError(f"Invalid inference configuration in {path}: {error}") from error


@dataclass(frozen=True)
class TranslationResult:
    text: str
    source_ids: tuple[int, ...]
    encoder_memory_shape: tuple[int, ...]
    initial_prefix: tuple[int, ...]
    hypothesis: Hypothesis


class TransformerTranslator:
    def __init__(
        self, model: EncoderDecoderTransformer, tokenizer: BilingualBPETokenizer, device: torch.device,
    ) -> None:
        if model.vocab_size != tokenizer.vocab_size or model.pad_id != tokenizer.pad_id:
            raise ValueError("Model and tokenizer are incompatible")
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, tokenizer_dir: Path, device: str = "auto",
    ) -> "TransformerTranslator":
        selected = select_device(device)
        model, tokenizer = load_translation_model(checkpoint, tokenizer_dir, selected)
        return cls(model, tokenizer, selected)

    def _source_ids(self, text: str, limit: int) -> tuple[int, ...]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("English input must be a nonempty string")
        ids = (*self.tokenizer.encode(text.strip()), self.tokenizer.eos_id)
        if len(ids) > limit:
            raise ValueError(f"Source needs {len(ids)} positions including EOS, exceeding limit {limit}")
        return ids

    def _next_logits(self, prefix: Tensor, source: Tensor, memory: Tensor) -> Tensor:
        hidden = self.model.decoder(
            prefix, memory, decoder_attention_mask(prefix, self.tokenizer.pad_id),
            cross_attention_mask(source, prefix.size(1), self.tokenizer.pad_id),
        )
        logits = self.model.output_projection(hidden[:, -1:, :])[:, 0, :]
        if logits.shape != (prefix.size(0), self.tokenizer.vocab_size):
            raise ValueError("Unexpected next-token vocabulary logits shape")
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite next-token logits at prefix length {prefix.size(1)}")
        return logits

    @torch.inference_mode()
    def generate(
        self, texts: Sequence[str], config: InferenceConfig | None = None, *, trace: bool = False,
    ) -> list[TranslationResult]:
        """Encode one batch and return ordered translations with optional step traces."""
        limit = self.model.config.max_seq_len
        config = config or InferenceConfig(max_new_tokens=min(64, limit))
        if isinstance(texts, str) or not texts:
            raise ValueError("Provide a nonempty sequence of English sentences")
        if config.max_new_tokens > limit:
            raise ValueError(f"max_new_tokens exceeds the model positional limit {limit}")
        tokenizer = self.tokenizer
        if config.strategy == "sampling" and config.top_k > tokenizer.vocab_size:
            raise ValueError("top_k exceeds the tokenizer vocabulary size")
        self.model.eval()
        sources = [self._source_ids(text, limit) for text in texts]
        source = torch.full((len(sources), max(map(len, sources))), tokenizer.pad_id, dtype=torch.long, device=self.device)
        for row, ids in enumerate(sources):
            source[row, :len(ids)] = source.new_tensor(ids)
        memory = self.model.encoder(source, encoder_attention_mask(source, tokenizer.pad_id))
        prefixes = source.new_full((len(sources), 1), tokenizer.bos_id)
        options = dict(eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id,
                       max_new_tokens=config.max_new_tokens, trace=trace)

        if config.strategy == "beam":
            hypotheses = []
            for row in range(len(sources)):
                source_row, memory_row = source[row:row + 1], memory[row:row + 1]

                def beam_logits(prefix: Tensor) -> Tensor:
                    return self._next_logits(prefix, source_row.expand(prefix.size(0), -1),
                                             memory_row.expand(prefix.size(0), -1, -1))

                hypotheses.append(beam_search_decode(
                    beam_logits, prefixes[row:row + 1], beam_width=config.beam_width,
                    length_penalty=config.length_penalty, **options,
                ))
        else:
            def next_logits(prefix: Tensor) -> Tensor:
                return self._next_logits(prefix, source, memory)

            if config.strategy == "greedy":
                hypotheses = greedy_decode(next_logits, prefixes, **options)
            else:
                rng = None if config.seed is None else torch.Generator().manual_seed(config.seed)

                def select(scores: Tensor) -> Tensor:
                    return sample_next_token(scores, temperature=config.temperature, top_k=config.top_k,
                                             top_p=config.top_p, generator=rng)

                hypotheses = decode_tokens(next_logits, prefixes, select=select, **options)

        controls = {tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id}
        return [
            TranslationResult(
                tokenizer.decode([token for token in hypothesis.token_ids if token not in controls],
                                 skip_special_tokens=False),
                ids, tuple(memory.shape), (tokenizer.bos_id,), hypothesis,
            )
            for ids, hypothesis in zip(sources, hypotheses)
        ]

    def translate(
        self, text: str, config: InferenceConfig | None = None, *, diagnostics: bool = False,
        trace: bool = False,
    ) -> str | TranslationResult:
        result = self.generate([text], config, trace=trace)[0]
        return result if diagnostics or trace else result.text
