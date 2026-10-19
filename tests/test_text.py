import pytest

from qxp_asr.text import normalize, scorer_normalize, strip_accents


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Irqikuna paykunapura qatipayanakunku.", "irqikuna paykunapura qatipayanakunku"),
        ("Hayk’aqmi runa Killaman chayarqan?", "hayk'aqmi runa killaman chayarqan"),
        ("{aa} Kusisqa kashani {eee} imatataq", "kusisqa kashani imatataq"),
        ("kay qhichwa siminchista [ ruido ].", "kay qhichwa siminchista"),
        ("Phisqa watapiqa  (na) allin llank'aq,", "phisqa watapiqa allin llank'aq"),
        ("Kaykuna kashan: nuqa, pay, qan,", "kaykuna kashan nuqa pay qan"),
        ("Ñuñoa llaqtata", "ñuñoa llaqtata"),
        ("Héroe Nacional", "heroe nacional"),
        ("'quoted' word", "quoted word"),
        ("90 años", "años"),
        (None, ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_keep_accents():
    assert normalize("paranqapaschá riki", keep_accents=True) == "paranqapaschá riki"


def test_strip_accents_keeps_enye():
    assert strip_accents("ÑAÑA ñaña áé") == "ÑAÑA ñaña ae"


def test_scorer_normalize_matches_organizer():
    assert scorer_normalize("¿Pipaqmi kay qillqasqa?") == "pipaqmi kay qillqasqa"
