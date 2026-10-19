from qxp_asr.combine import EPS, confusion_network, medoid, rover


def test_medoid_picks_consensus():
    hyps = ["a b c d", "a b x d", "a b c d", "q r s"]
    assert medoid(hyps) == "a b c d"


def test_rover_majority_substitution():
    assert rover(["a b c", "a x c", "a x c"]) == "a x c"


def test_rover_handles_insertions_and_deletions():
    # system 2 inserts "z", systems 1 and 3 do not -> dropped; system 3 deletes "c" -> kept (2:1)
    assert rover(["a b c", "a z b c", "a b"]) == "a b c"


def test_rover_weights_break_ties():
    assert rover(["a b", "a c"], weights=[1.0, 2.0]) == "a c"


def test_rover_identity():
    assert rover(["kay wasi", "kay wasi"]) == "kay wasi"


def test_confusion_network_tracks_votes():
    cn = confusion_network(["a b c", "a x c", "a c"])
    assert len(cn) == 3
    assert cn[0]["a"] == {0, 1, 2}
    assert cn[1]["b"] == {0} and cn[1]["x"] == {1} and cn[1][EPS] == {2}
    # every hypothesis occupies exactly one entry per slot
    for slot in cn:
        assert sorted(i for v in slot.values() for i in v) == [0, 1, 2]


def test_confusion_network_insertion_slot():
    cn = confusion_network(["a c", "a b c"])
    assert len(cn) == 3
    assert cn[1]["b"] == {1} and cn[1][EPS] == {0}
