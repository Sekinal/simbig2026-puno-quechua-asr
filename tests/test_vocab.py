from qxp_asr.vocab import SYMBOLS, ctc_collapse, decode, encode


def test_roundtrip():
    text = "hayk'aqmi runa ñawpaq"
    assert decode(encode(text)) == text


def test_ctc_collapse_removes_repeats_and_blanks():
    a, b = SYMBOLS.index("a"), SYMBOLS.index("b")
    assert ctc_collapse([0, a, a, 0, a, b, b, 0]) == [a, a, b]


def test_vocab_size():
    assert len(SYMBOLS) == 30
