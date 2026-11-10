from pathlib import Path
from streamlit.testing.v1 import AppTest
from checkpoint_02.corpus import ROOT


def test_app_start_and_controls():
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=45).run()
    assert not app.exception
    assert [m.value for m in app.metric[:3]]==['3','105','90']
    app.selectbox(key='strategy').select('sentence').run()
    app.select_slider(key='size').set_value(128).run()
    app.selectbox(key='mode').select('stem').run()
    assert not app.exception
    scope=app.selectbox(key='scope')
    scope.select(scope.options[2]).run()
    assert not app.exception and len(app.selectbox(key='chunk').options)>0
    app.text_input(key='query').set_value('zzzz_nonexistent_token_zzzz').run()
    assert not app.exception
    app.button(key='duplicates').click().run()
    assert not app.exception
