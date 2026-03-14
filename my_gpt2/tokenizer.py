import json
import re
from functools import lru_cache
import regex # GPT-2の正規表現は\p{L}を使うため'regex'ライブラリが必要

@lru_cache()
def bytes_to_unicode():
    """
    UTF-8バイト列と対応するユニコード文字列のリストを返す。
    GPT-2トークナイザーでバイトを文字列にマッピングするために使用する。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """単語中のシンボルペアの集合を返す。"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Tokenizer:
    def __init__(self, vocab_path="weights/vocab.json", merges_path="weights/merges.txt"):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            bpe_data = f.read().split("\n")[1:-1]
            self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(bpe_data)}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # GPT-2固有の正規表現
        self.pat = regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", regex.IGNORECASE)

    def bpe(self, token):
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token

        while True:
            # merges.txtに基づいてランクが最小（最頻出）のペアを探す
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        return " ".join(word)

    def encode(self, text):
        bpe_tokens = []
        # 正規表現で事前トークン化
        for token in regex.findall(self.pat, text):
            # バイトを特殊文字に変換
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            # BPEマージルールを適用
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        # IDを特殊文字に変換
        text = "".join([self.decoder[token] for token in tokens])
        # 特殊文字を元のバイト列に変換
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")
        return text
