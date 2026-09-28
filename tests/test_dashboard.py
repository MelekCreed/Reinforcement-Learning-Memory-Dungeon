from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_dashboard_step_reset_and_pair():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"dashboard/app.py"),default_timeout=180).run()
    assert not app.exception
    next(b for b in app.button if b.label == "→ Step").click().run()
    assert not app.exception
    assert app.session_state["session"].env.steps == 1
    next(b for b in app.button if b.label == "Fork & compare from here").click().run()
    assert not app.exception
    assert len(app.session_state["pair"]) == 2
    next(b for b in app.button if b.label == "↻ Reset episode").click().run()
    assert not app.exception
    assert app.session_state["session"].env.steps == 0
    old_seed = app.session_state["session"].env.seed
    next(b for b in app.button if b.label == "＋ New dungeon").click().run()
    assert not app.exception
    assert app.session_state["session"].env.seed == old_seed+1
