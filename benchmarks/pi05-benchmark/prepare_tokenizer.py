"""Create and validate Transformers 5 JSON from the checkpoint's SentencePiece model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import sentencepiece
from tokenizers import processors
from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import GemmaConverter, GemmaSentencePieceExtractor


class SentencePieceExtractorWithProcessor(GemmaSentencePieceExtractor):
    """Supply the processor still required by the pinned Transformers 5.3 converter."""

    def __init__(self, path: str):
        super().__init__(path)
        self.sp = sentencepiece.SentencePieceProcessor(model_file=path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="Local cache snapshot with tokenizer.model and tokenizer_config.json",
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="LIBERO libero_10 directory")
    parser.add_argument("--output", type=Path, required=True, help="New JSON validation report")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = args.tokenizer_dir / "tokenizer.model"
    sp = sentencepiece.SentencePieceProcessor(model_file=str(model))
    if sp.vocab_size() != 257152:
        raise ValueError("expected the PI0.5 checkpoint's 257152-token SentencePiece vocabulary")
    tasks = sorted(args.dataset_root.glob("*.hdf5"))[:10]
    if len(tasks) != 10:
        raise ValueError("expected at least 10 LIBERO task files")
    texts = [
        "pick up the black bowl",
        "open the drawer",
        "close the drawer",
        "turn on the stove",
        "Task: pick up the black bowl, State: 1 2 3;\nAction: ",
        "  double  space\n",
        "put the bowl on the plate.",
        "抓起杯子",
        "123 0 -1 255",
    ]
    for path in tasks:
        with h5py.File(path) as handle:
            task = json.loads(handle["data"].attrs["problem_info"])["language_instruction"]
        texts.append(task)
        for j in range(10):
            state = " ".join(str((i * 17 + j) % 256) for i in range(32))
            texts.append(f"Task: {task}, State: {state};\nAction: ")
    expected = {text: sp.encode(text, add_bos=True) for text in texts}
    tokenizer_json = args.tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        converter = GemmaConverter(
            SimpleNamespace(
                vocab_file=str(model),
                pad_token="<pad>",
                eos_token="<eos>",
                bos_token="<bos>",
                add_prefix_space=False,
            )
        )
        converter.SpmExtractor = SentencePieceExtractorWithProcessor
        backend = converter.converted()
        backend.post_processor = processors.TemplateProcessing(
            single="<bos> $A",
            pair="<bos> $A <bos> $B:1",
            special_tokens=[("<bos>", sp.bos_id())],
        )
        for text, ids in expected.items():
            if backend.encode(text).ids != ids:
                raise ValueError(f"converted token IDs differ from SentencePiece: {text!r}")
        backend.save(str(tokenizer_json))
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
    rows = []
    for text in texts:
        actual = tokenizer.encode(text)
        if actual != expected[text]:
            raise ValueError(f"loaded token IDs differ from SentencePiece: {text!r}")
        rows.append({"text": text, "ids": actual})
    if len(tokenizer) != sp.vocab_size():
        raise ValueError("loaded tokenizer vocabulary differs from SentencePiece")
    report = {
        "passed": True,
        "calls": len(rows),
        "corrected_vocab_size": len(tokenizer),
        "scope": "Exact text token IDs; LIBERO instructions, state-text probes, whitespace and Unicode",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "tokenizer_json_sha256": hashlib.sha256(tokenizer_json.read_bytes()).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "tokenizers", "sentencepiece", "protobuf")
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Validated {len(rows)} inputs; vocabulary={len(tokenizer)}; {args.output}")


if __name__ == "__main__":
    main()
