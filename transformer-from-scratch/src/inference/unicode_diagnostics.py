"""Read-only views of ByteLevel pieces and decoded Unicode."""

import unicodedata

from ..data.tokenizer import BilingualBPETokenizer


def describe_text(text: str) -> dict:
    return {
        "text": text, "escaped": text.encode("unicode_escape").decode("ascii"),
        "nfc": unicodedata.is_normalized("NFC", text),
        "replacement_characters": text.count("\ufffd"),
        "controls": [f"U+{ord(char):04X}" for char in text
                     if unicodedata.category(char) in ("Cc", "Cf") and char not in "\u200c\u200d"],
        "code_points": [{"code": f"U+{ord(char):04X}", "name": unicodedata.name(char, "UNNAMED"),
                         "category": unicodedata.category(char)} for char in text],
    }


def bytelevel_bytes(token_strings: list[str]) -> bytes:
    """Reverse the ByteLevel visible-byte alphabet for non-special pieces."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    extra = [value for value in range(256) if value not in visible]
    inverse = dict(zip(
        map(chr, visible + list(range(256, 256 + len(extra)))), visible + extra, strict=True,
    ))
    return bytes(inverse[char] for piece in token_strings for char in piece)


def describe_ids(tokenizer: BilingualBPETokenizer, ids: list[int]) -> dict:
    controls = {tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id}
    decoded_ids = [token for token in ids if token not in controls]
    text = tokenizer.decode(decoded_ids, skip_special_tokens=False)
    raw = bytearray()
    for token in decoded_ids:
        if token == tokenizer.unk_id:
            raw.extend(b"<unk>")
        else:
            raw.extend(bytelevel_bytes([tokenizer.id_to_token(token)]))
    try:
        raw.decode("utf-8", errors="strict")
        utf8_valid = True
    except UnicodeDecodeError:
        utf8_valid = False
    return {
        "ids": ids, "token_strings": [tokenizer.id_to_token(token) for token in ids],
        "bytes_hex": raw.hex(), "strict_utf8_valid": utf8_valid,
        "bytelevel_agrees_with_decoder": raw.decode("utf-8", errors="replace") == text,
        **describe_text(text),
    }
