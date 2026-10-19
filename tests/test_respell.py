import pytest

from qxp_asr.respell import Respeller, align_words, is_spanish_like, learn_maps, respell


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Puno stem forms (README conventions).
        ("Huk allqu", "huq alqu"),
        ("askha qullqi", "ashkha qulqi"),
        ("iskay chakrapi", "ishkay chaqrapi"),
        ("aswan chaqay", "ashwan chhaqay"),
        ("llamk'anchis qillqasqata", "llank'anchis qhilqasqata"),
        ("misk'i maskhaspa", "mishk'i mashkhaspa"),
        # Cusco 5-vowel spelling next to q -> 3 vowels; whole word lowered.
        ("Ñoqaqa Qosqo llaqtapim tiyani.", "nuqaqa qusqu llaqtapin tiyani"),
        ("mosoq sonqo qhepa qolqe", "musuq sunqu qhipa qulqi"),
        # Ayacucho morphology: -chka- -> -sha-, -chik -> -chis, -m evidential -> -n.
        ("rimachkanku kanchik", "rimashanku kanchis"),
        ("ñuqanchikpa wasimpi", "nuqanchispa wasinpi"),
        ("chaypim aswampis", "chaypin ashwanpis"),
        # Diphthongs written with glides.
        ("kausay ñaupaq", "kawsay ñawpaq"),
    ],
)
def test_respell_rules(raw, expected):
    assert respell(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "pedroqa manuelqa rosaspataman",  # names with Quechua suffixes keep e/o
        "pampapi hampi llamphu",  # stems that keep mp in the curated lexicon
        "juliaca sicuani quispe",  # Spanish spellings
        "tecnologia alcalde",
        "yachachik",  # agentive -chik is not the 1pl.incl suffix
        "kampupi",  # in-lexicon loan
    ],
)
def test_respell_leaves_loans_and_names(text):
    assert respell(text) == text


def test_mp_needs_known_host_word():
    # urupampa / qaqatampu are place names: m stays (host word unknown or mp stem).
    assert respell("urupampa qaqatampu") == "urupampa qaqatampu"
    assert respell("llaqtampi") == "llaqtanpi"


def test_hina_and_rayku_joining():
    assert respell("chay hina") == "chayhina"
    assert respell("parlaspa hina") == "parlaspahina"
    assert respell("rikuykun hina") == "rikuykun hina"  # finite verb: stays apart
    assert respell("manachuhina") == "manachu hina"
    assert respell("runanchis rayku") == "runanchisrayku"
    assert respell("hinaqa hinaspa") == "hinaqa hinaspa"


def test_loanword_evidential():
    assert respell("distritum") == "distritun"
    assert respell("album") == "album"


def test_is_spanish_like():
    assert is_spanish_like("pedro")
    assert is_spanish_like("quispe")
    assert not is_spanish_like("chaqra")
    assert not is_spanish_like("qhichwa")


def test_respell_is_idempotent():
    text = "Huk allqum wasimpi kachkan, ñuqanchik qosqopi"
    once = respell(text)
    assert respell(once) == once


def test_learn_maps_and_from_pairs():
    pairs = [
        ("huk wasi", "huq wasi"),
        ("huk runa", "huq runa"),
        ("yachaywasipi kani", "yachay wasipi kani"),
        ("tayta mamay", "taytamamay"),
    ]
    words, phrases = learn_maps(pairs)
    assert words["huk"] == "huq"
    assert words["yachaywasipi"] == "yachay wasipi"
    assert phrases[("tayta", "mamay")] == ("taytamamay",)
    r = Respeller.from_pairs(pairs)
    assert r("Tayta mamay huk") == "taytamamay huq"
    assert "wasi" in r.lexicon


def test_learn_maps_respects_majority():
    # "pay" is kept twice and changed once: no mapping.
    words, _ = learn_maps([("pay", "pay"), ("pay", "pay"), ("pay", "paymi")])
    assert "pay" not in words


def test_align_words():
    ops = align_words(["a", "b", "c"], ["a", "x", "c"])
    assert ("replace", ("b",), ("x",)) in ops
