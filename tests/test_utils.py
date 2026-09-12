import os


def _lazy_import():
    global available_models, download_model
    from faster_whisper import available_models as __am
    from faster_whisper import download_model as __dm

    available_models = __am
    download_model = __dm


def _available_models_proxy(*args, **kwargs):
    _lazy_import()
    return available_models(*args, **kwargs)


def _download_model_proxy(*args, **kwargs):
    _lazy_import()
    return download_model(*args, **kwargs)


available_models = _available_models_proxy
download_model = _download_model_proxy


def test_available_models():
    models = available_models()
    assert isinstance(models, list)
    assert "tiny" in models


def test_download_model(tmpdir):
    output_dir = str(tmpdir.join("model"))

    model_dir = download_model("tiny", output_dir=output_dir)

    assert model_dir == output_dir
    assert os.path.isdir(model_dir)
    assert not os.path.islink(model_dir)

    for filename in os.listdir(model_dir):
        path = os.path.join(model_dir, filename)
        assert not os.path.islink(path)


def test_download_model_in_cache(tmpdir):
    cache_dir = str(tmpdir.join("model"))
    download_model("tiny", cache_dir=cache_dir)
    assert os.path.isdir(cache_dir)
