import pytest
from my_gpt2.spiece import SentencePieceTokenizer

# 期待値は sentencepiece ライブラリで生成
#   import sentencepiece as spm
#   sp = spm.SentencePieceProcessor(); sp.Load("weights/rinna/japanese-gpt2-small/spiece.model")
#   sp.EncodeAsIds(text), sp.EncodeAsPieces(text), sp.DecodeIds(ids)

@pytest.fixture(scope="module")
def tokenizer():
    return SentencePieceTokenizer("rinna/japanese-gpt2-small")


def test_special_token_ids(tokenizer):
    assert tokenizer.unk_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2


def test_encode_japanese(tokenizer):
    # '吾輩は猫である。' → ['▁', '吾', '輩', 'は', '猫', 'である', '。']
    assert tokenizer.encode("吾輩は猫である。") == [9, 5361, 31082, 11, 4324, 27, 8]


def test_encode_greeting(tokenizer):
    # 'こんにちは' → ['▁', 'こんにち', 'は']
    assert tokenizer.encode("こんにちは") == [9, 30442, 11]


def test_encode_mixed(tokenizer):
    # '日本語と English の混在'
    # 'E' は語彙外なので UNK (id=0)
    assert tokenizer.encode("日本語と English の混在") == [9, 2481, 20, 9, 0, 6099, 391, 15713, 9, 10, 27190]



def test_decode_japanese(tokenizer):
    ids = [9, 5361, 31082, 11, 4324, 27, 8]
    assert tokenizer.decode(ids) == "吾輩は猫である。"


def test_decode_greeting(tokenizer):
    ids = [9, 30442, 11]
    assert tokenizer.decode(ids) == "こんにちは"


def test_roundtrip(tokenizer):
    for text in ["吾輩は猫である。", "こんにちは", "日本語テスト"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text
