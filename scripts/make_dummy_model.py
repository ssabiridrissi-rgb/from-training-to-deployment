"""Crée un FAUX modèle minuscule (quelques Ko) avec la même interface que DistilBERT.

Il sert uniquement aux tests automatiques (pytest + CI GitHub Actions), pour ne pas
dépendre du vrai modèle de ~250 Mo. Il "compte" les mots positifs et négatifs.

    python scripts/make_dummy_model.py      (écrit dans models/)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

POS = ["magnifique", "adoré", "excellent", "génial", "superbe", "bon", "drôle"]
NEG = ["ennuyeux", "décevant", "nul", "mauvais", "horrible", "raté", "lent"]
CLS, SEP = "[CLS]", "[SEP]"
# Quand on lance le script, il écrit toujours dans models/ (dossier fixe, pas d'argument).
# Les tests, eux, appellent build() directement avec un dossier temporaire.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    vocab = {"[PAD]": 0, "[UNK]": 1, CLS: 2, SEP: 3}
    for w in POS + NEG + ["ce", "film", "est", "un", "le", "la"]:
        vocab.setdefault(w, len(vocab))

    # Tokenizer (même format tokenizer.json que les vrais tokenizers Hugging Face)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(
        single=f"{CLS} $A {SEP}", special_tokens=[(CLS, 2), (SEP, 3)]
    )
    tk.save(str(out / "tokenizer.json"))

    # Table : chaque mot vote [négatif, positif]
    table = np.zeros((len(vocab), 2), dtype=np.float32)
    for w in POS:
        table[vocab[w]] = [0.0, 2.0]
    for w in NEG:
        table[vocab[w]] = [2.0, 0.0]
    table[vocab[CLS]] = [0.1, 0.0]  # léger biais négatif si aucun mot connu

    # Graphe ONNX : logits = somme(table[input_ids] * attention_mask)
    nodes = [
        helper.make_node("Gather", ["table", "input_ids"], ["emb"]),
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask_f", "axis_last"], ["mask_3d"]),
        helper.make_node("Mul", ["emb", "mask_3d"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "axis_seq"], ["logits"], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes, "dummy_sentiment",
        inputs=[helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
                helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
        initializer=[numpy_helper.from_array(table, "table"),
                     numpy_helper.from_array(np.array([-1], dtype=np.int64), "axis_last"),
                     numpy_helper.from_array(np.array([1], dtype=np.int64), "axis_seq")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(out / "model.onnx"))

    (out / "config.json").write_text(json.dumps({
        "labels": ["négatif", "positif"],
        "max_length": 16,
        "version": "dummy-test",
        "examples": ["Ce film est magnifique", "Un film ennuyeux et raté"],
        "reference": {"mean_confidence": 0.95, "mean_tokens": 8},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build(MODELS_DIR))
