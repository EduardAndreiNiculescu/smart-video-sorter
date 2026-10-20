from pathlib import Path

from svs import catalog
from svs.broll import shot_type
from svs.captions import build_ass, chunk_words, correct_word
from svs.shorts import slugify, widen_window
from svs.transcribe import Segment, mini_subject, score_highlights
from svs.vision import VisionEngine


def test_event_name_from_folder_depth(tmp_path):
    root = tmp_path / "raw"
    clip = root / "2024 - Annual Gala" / "CamA" / "c.MTS"
    assert catalog.derive_event_name(clip, root) == "2024 - Annual Gala"
    assert catalog.derive_event_name(root / "Operator" / "Expo: day 1" / "x.mov", root, depth=2) == "Expo_ day 1"
    assert catalog.derive_event_name(root / "loose.mp4", root) == "Unsorted"


def test_widen_window_hits_min_length_and_stays_in_bounds():
    assert widen_window(10, 14, 120) == (4.5, 19.5)          # 4 s -> 15 s, centred
    assert widen_window(1, 3, 120) == (0.0, 15.0)            # shifted right at the start
    assert widen_window(110, 118, 120) == (105.0, 120.0)     # shifted left at the end
    assert widen_window(0, 90, 120) == (0, 90)               # long enough already


def test_slugify_keeps_romanian_letters():
    assert slugify("În 2019 am fondat clubul!") == "În_2019_am_fondat_clubul"
    assert slugify("???") == "highlight"


def test_highlight_scoring():
    segs = [
        Segment(0, 5, "Care e cea mai mare greșeală în marketing?", -0.2),  # question + keyword
        Segment(5, 6, "Da!", -0.1),                                          # too short
        Segment(6, 12, "Vremea a fost frumoasă ieri", -0.3),                # nothing
        Segment(12, 20, "Profitul a crescut după strategie", -0.2),         # 2 keywords
    ]
    hl = score_highlights(segs, keywords={"marketing", "profit", "strategie"})
    assert [h["start"] for h in hl] == [12, 0]
    assert hl[0]["score"] == 6 and hl[1]["reason"] == "Question"


def test_mini_subject():
    assert mini_subject("Bună seara tuturor. Azi vorbim despre bani") == "Bună_seara_tuturor"


def test_vision_caption_mapping():
    assert VisionEngine._categorize_caption("A man typing on a computer screen in an office") == "Screen_Recording"
    assert VisionEngine._categorize_caption("A tennis court outdoors under a blue sky") == "Outdoor"
    assert VisionEngine._categorize_caption("blurry") is None


def test_event_shot_types():
    assert shot_type(["A speaker at a podium with a microphone"]) == "B-ROLL:Stage"
    assert shot_type(["people receiving a trophy on stage"]) == "B-ROLL:Awards"
    assert shot_type(["a dog"]) == "B-ROLL:Event_Misc"


def test_karaoke_captions():
    words = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, w in enumerate("one two three. four five six seven eight".split())]
    chunks = chunk_words(words, max_words=5)
    assert [len(c) for c in chunks] == [3, 5]
    ass = build_ass(words)
    assert ass.count("Dialogue:") == 8
    assert r"{\c&H00FF00&}one {\c&HFFFFFF&}two" in ass
    assert correct_word("crayova,", {"crayova": "Craiova"}) == "Craiova,"


def test_catalog_resume_and_export(tmp_path):
    conn = catalog.init_db(tmp_path / "video_catalog.db")
    src = tmp_path / "a.mp4"
    assert not catalog.is_processed(conn, src)
    catalog.record_entry(conn, catalog.CatalogEntry(str(src), "done", category="B-ROLL:Outdoor",
                                                    dest_path=str(tmp_path / "E" / "a.mp4"), duration=75, event="E"))
    assert catalog.is_processed(conn, src)
    out = tmp_path / "catalog.csv"
    catalog.export_editor_csv(conn, out)
    text = out.read_text(encoding="utf-8-sig")
    assert "B-Roll: Outdoor" in text and "01:15" in text


def test_unique_destination_never_overwrites(tmp_path):
    (tmp_path / "x.mp4").write_bytes(b"1")
    assert catalog.unique_destination(tmp_path, "x.mp4").name == "x_1.mp4"
