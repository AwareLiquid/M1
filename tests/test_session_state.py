from mt_lnn.session_state import HFSessionState, load_session, save_session


def test_session_roundtrip(tmp_path):
    path = tmp_path / "s.json"
    s = HFSessionState(session_id="abc")
    s.open_questions.append("what is m-theory")
    s.evidence_log.append({"source": "mock", "query": "q", "fact_len": 12})
    s.history.append({"role": "user", "text": "hi"})

    save_session(s, str(path))
    loaded = load_session(str(path))

    assert loaded.session_id == "abc"
    assert loaded.open_questions == ["what is m-theory"]
    assert loaded.evidence_log[0]["source"] == "mock"
    assert loaded.history[0]["role"] == "user"


def test_session_defaults_empty():
    s = HFSessionState(session_id="x")
    assert s.open_questions == []
    assert s.evidence_log == []
    assert s.history == []
