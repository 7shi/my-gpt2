import math
import struct
import unicodedata


def _read_varint(data, pos):
    """LEB128 形式の可変長整数をデコードする。"""
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _parse_fields(data, start, end):
    """Protobuf フィールドを start〜end の範囲で順に yield する。
    wire_type: 0=varint, 1=64-bit, 2=length-delimited, 5=float32
    """
    pos = start
    while pos < end:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:
            val, pos = _read_varint(data, pos)
            yield field_num, wire_type, val
        elif wire_type == 1:
            val = struct.unpack_from('<Q', data, pos)[0]; pos += 8
            yield field_num, wire_type, val
        elif wire_type == 2:
            length, pos = _read_varint(data, pos)
            val = data[pos:pos + length]; pos += length
            yield field_num, wire_type, val
        elif wire_type == 5:
            val = struct.unpack_from('<f', data, pos)[0]; pos += 4
            yield field_num, wire_type, val
        else:
            raise ValueError(f"未対応の wire_type={wire_type} at pos={pos}")


def _load_vocab(model_path):
    """spiece.model の ModelProto を解析する。
    戻り値: (vocab, normalizer_name)
      vocab: list of (piece: str, score: float, type: int)
      normalizer_name: str（例: 'nmt_nfkc'）または None
    """
    data = open(model_path, "rb").read()
    vocab = []
    normalizer_name = None
    for fnum, wtype, val in _parse_fields(data, 0, len(data)):
        if fnum == 1 and wtype == 2:  # ModelProto.pieces
            piece, score, ptype = None, 0.0, 1
            for f2, w2, v2 in _parse_fields(val, 0, len(val)):
                if f2 == 1 and w2 == 2:
                    piece = v2.decode("utf-8")
                elif f2 == 2 and w2 == 5:
                    score = v2  # float32
                elif f2 == 3 and w2 == 0:
                    ptype = v2
            vocab.append((piece, score, ptype))
        elif fnum == 3 and wtype == 2:  # normalizer_spec
            for f2, w2, v2 in _parse_fields(val, 0, len(val)):
                if f2 == 1 and w2 == 2:  # normalizer_spec.name
                    normalizer_name = v2.decode("utf-8")
    return vocab, normalizer_name


def _escape_piece(piece):
    """制御文字を <0xXX> 形式にエスケープする。"""
    return "".join(
        f"<0x{ord(c):02X}>" if ord(c) < 0x20 or ord(c) == 0x7F else c
        for c in piece
    )


def save_vocab(model_path, output_path=None):
    """spiece.model の語彙を .vocab 形式（ピース\tスコア）で保存する。
    output_path を省略すると model_path の拡張子を .vocab に変えたパスを使う。
    制御文字は <0xXX> 形式にエスケープする。
    """
    if output_path is None:
        output_path = model_path.rsplit(".", 1)[0] + ".vocab"
    vocab, _ = _load_vocab(model_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for piece, score, _ in vocab:
            f.write(f"{_escape_piece(piece)}\t{score:.6f}\n")
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="spiece.model を .vocab 形式に変換する")
    parser.add_argument("model", help="spiece.model のパス")
    parser.add_argument("-o", "--output", default=None, help="出力先ファイルパス（省略時は拡張子を .vocab に変換）")
    args = parser.parse_args()
    out = save_vocab(args.model, args.output)
    print(out)


class SentencePieceTokenizer:
    def __init__(self, model_id="rinna/japanese-gpt2-small"):
        path = f"weights/{model_id}/spiece.model"
        vocab, self._normalizer = _load_vocab(path)
        self._id_to_piece = [piece for piece, score, ptype in vocab]
        self._piece_to_id = {piece: i for i, (piece, score, ptype) in enumerate(vocab)}
        self._piece_to_score = {piece: score for piece, score, ptype in vocab}
        self.unk_id = self._piece_to_id.get("<unk>", 0)
        self.bos_id = self._piece_to_id.get("<s>", 1)
        self.eos_id = self._piece_to_id.get("</s>", 2)

    def _normalize(self, text):
        # normalizer_spec.name に応じたテキスト変換
        if self._normalizer and "nfkc" in self._normalizer:
            text = unicodedata.normalize("NFKC", text)
        # スペースを ▁ に置換して先頭に ▁ を付加
        return "▁" + text.replace(" ", "▁")

    def encode(self, text):
        """テキストを Viterbi アルゴリズムでトークン ID 列に変換する。"""
        normalized = self._normalize(text)
        n = len(normalized)

        # best[i] = (累積スコア, 前の pos, piece)
        best = [(-math.inf, -1, None)] * (n + 1)
        best[0] = (0.0, -1, None)

        for i in range(n):
            if best[i][0] == -math.inf:
                continue
            for j in range(i + 1, n + 1):
                piece = normalized[i:j]
                if piece in self._piece_to_score:
                    score = best[i][0] + self._piece_to_score[piece]
                    if score > best[j][0]:
                        best[j] = (score, i, piece)
                elif j == i + 1:
                    # 単一文字で語彙外の場合は UNK として扱う
                    score = best[i][0] + (-1e10)
                    if score > best[j][0]:
                        best[j] = (score, i, "<unk>")

        # バックトラックで最適ピース列を復元
        pieces = []
        pos = n
        while pos > 0:
            _, prev, piece = best[pos]
            pieces.append(piece)
            pos = prev
        pieces.reverse()

        return [self._piece_to_id.get(p, self.unk_id) for p in pieces]

    def decode(self, tokens):
        """トークン ID 列をテキストに変換する。"""
        pieces = [self._id_to_piece[i] for i in tokens]
        text = "".join(pieces)
        return text.replace("▁", " ").lstrip(" ")
