"""
Tests for the fiftyone/utils/videodb.py VideoDB integration.

The tests use a mocked VideoDB connection, so no API key or network access is
required.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from unittest import mock

import pytest

import fiftyone as fo  # pylint: disable=unused-import
import fiftyone.core.labels as fol
import fiftyone.core.metadata as fome
import fiftyone.utils.videodb as fouv

pytest.importorskip("videodb")

_VIDEO_URL = "https://example.com/video.mp4"

_METADATA = fome.VideoMetadata(
    total_frame_count=300,
    frame_rate=30.0,
    frame_width=640,
    frame_height=480,
    duration=10.0,
)


def _config(**kwargs):
    return fouv.VideoDBModelConfig(kwargs)


def _analyzer(status="done", output=None):
    analyzer = mock.MagicMock()
    analyzer.status = status
    analyzer.is_successful = status == "done"
    analyzer.get_output.return_value = output
    return analyzer


@pytest.fixture(autouse=True)
def clear_uploads():
    fouv._UPLOADS.clear()
    yield
    fouv._UPLOADS.clear()


@pytest.fixture
def video():
    video = mock.MagicMock()
    video.understand.return_value.id = "understanding-1"
    return video


@pytest.fixture
def conn(video):
    conn = mock.MagicMock()
    conn.get_collection.return_value.upload.return_value = video
    return conn


def _model(conn, **kwargs):
    kwargs.setdefault("api_key", "test-key")
    model = fouv.VideoDBModel(_config(**kwargs))
    model._conn = conn
    return model


def _predict(model, video, output, path=_VIDEO_URL):
    analyzer = _analyzer(output=output)
    video.understand.return_value.get_analyzer.return_value = analyzer
    with mock.patch.object(
        model, "_get_video_metadata", return_value=_METADATA
    ):
        return model.predict(path)


class TestVideoDBModelConfig:
    def test_defaults(self):
        config = _config()

        assert config.analyzer == "spoken_words"
        assert config.api_key is None
        assert config.collection_id is None
        assert config.prompt is None
        assert config.inputs is None
        assert config.sampling is None
        assert config.segmentation is None
        assert config.transform is None
        assert config.timeout == 1800
        assert config.poll_interval == 10

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"analyzer": "embed"}, "Unsupported analyzer"),
            ({"prompt": "What?"}, "prompt is not supported"),
            ({"model": "pro"}, "model is not supported"),
            ({"sampling": {"frame_count": 1}}, "sampling is not supported"),
            ({"analyzer": "ocr", "language": "en"}, "language is not"),
            ({"analyzer": "ocr", "inputs": ["ocr"]}, "inputs is not"),
            ({"analyzer": "vlm", "sandbox_id": "sb-1"}, "sandbox_id is not"),
            ({"analyzer": "vlm", "inputs": ["faces"]}, "Unsupported input"),
        ],
    )
    def test_invalid_values_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _config(**kwargs)


class TestVideoDBModel:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("VIDEO_DB_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No VideoDB API key"):
            fouv.VideoDBModel(_config())

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("VIDEO_DB_API_KEY", "env-key")
        with mock.patch.object(fouv.videodb, "connect") as connect:
            fouv.VideoDBModel(_config())

        connect.assert_called_once_with(api_key="env-key")

    def test_transcript(self, conn, video):
        output = {
            "scenes": [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "data": {
                        "text": "Hello there.",
                        "words": [{"text": "Hello", "speaker": "A"}],
                    },
                },
                {"start": 4.0, "end": 5.0, "data": {"text": ""}},
                {"start": 5.0, "end": 10.0, "data": {"text": "Goodbye."}},
            ]
        }
        model = _model(conn, language="en")

        label = _predict(model, video, output)

        assert isinstance(label, fol.TemporalDetections)
        first, second = label.detections
        assert first.label == "transcript"
        assert first.support == [1, 121]
        assert first["text"] == "Hello there."
        assert first["words"][0]["speaker"] == "A"
        assert second.support == [151, 300]

        conn.get_collection.assert_called_once_with("default")
        upload = conn.get_collection.return_value.upload
        upload.assert_called_once_with(source=_VIDEO_URL)

        kwargs = video.understand.call_args.kwargs
        assert kwargs["analyzers"] == [
            {
                "type": "spoken_words",
                "name": "transcript",
                "config": {"language": "en"},
            }
        ]
        assert kwargs["segmentation"] is None
        assert kwargs["transform"] is None

        understanding = video.understand.return_value
        understanding.wait_until_complete.assert_called_once_with(
            timeout=1800, poll_interval=10
        )
        understanding.get_analyzer.assert_called_once_with(
            "transcript", refresh=True
        )

    def test_scene_descriptions(self, conn, video):
        output = {
            "scenes": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "data": {"activity": "running", "label": "reserved"},
                }
            ]
        }
        model = _model(
            conn,
            analyzer="vlm",
            collection_id="c-123",
            prompt="Describe it. Text: {{inputs.ocr}}",
            schema={"activity": "string"},
            inputs=["ocr", {"type": "spoken_words", "name": "speech"}],
            model="pro",
            sampling={"strategy": "uniform", "frame_count": 4},
            segmentation={"type": "time", "seconds": 5},
            transform={"resolution": "480p"},
            timeout=60,
            poll_interval=1,
        )

        label = _predict(model, video, output)

        (detection,) = label.detections
        assert detection.label == "scene"
        assert detection.support == [1, 300]
        assert detection["activity"] == "running"
        assert detection["label_"] == "reserved"

        conn.get_collection.assert_called_once_with("c-123")

        kwargs = video.understand.call_args.kwargs
        assert kwargs["analyzers"] == [
            {"type": "ocr", "name": "ocr"},
            {"type": "spoken_words", "name": "speech"},
            {
                "type": "vlm",
                "name": "scene",
                "inputs": ["ocr", "speech"],
                "sampling": {"strategy": "uniform", "frame_count": 4},
                "config": {
                    "prompt": "Describe it. Text: {{inputs.ocr}}",
                    "schema": {"activity": "string"},
                    "model": "pro",
                },
            },
        ]
        assert kwargs["segmentation"] == {"type": "time", "seconds": 5}
        assert kwargs["transform"] == {"resolution": "480p"}

        understanding = video.understand.return_value
        understanding.wait_until_complete.assert_called_once_with(
            timeout=60, poll_interval=1
        )
        understanding.get_analyzer.assert_called_once_with(
            "scene", refresh=True
        )

    @pytest.mark.parametrize(
        "analyzer,name,data",
        [
            ("ocr", "ocr", {"text": "EXIT"}),
            (
                "brand_detection",
                "brands",
                {"brand_names": ["Nike"], "mentions": []},
            ),
            ("activity_recognition", "activity", {"activity": "walking"}),
            ("location_detection", "location", {"location": "street"}),
        ],
    )
    def test_vlm_backed_analyzers(self, conn, video, analyzer, name, data):
        output = {"scenes": [{"start": 0.0, "end": 2.0, "data": data}]}
        model = _model(conn, analyzer=analyzer, model="mini")

        label = _predict(model, video, output)

        (detection,) = label.detections
        assert detection.label == name
        for key, value in data.items():
            assert detection[key] == value

        kwargs = video.understand.call_args.kwargs
        assert kwargs["analyzers"] == [
            {"type": analyzer, "name": name, "config": {"model": "mini"}}
        ]

    def test_object_detection(self, conn, video):
        output = {
            "scenes": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "data": {
                        "frames": [
                            {"timestamp": 0.0, "detections": []},
                            {
                                "timestamp": 1.0,
                                "detections": [
                                    {
                                        "label": "stop sign",
                                        "score": 0.5,
                                        "box": {
                                            "box": [0.25, 0.5, 0.75, 1.0],
                                            "unit": "normalized",
                                        },
                                    },
                                    {
                                        "label": "car",
                                        "score": 0.9,
                                        "box": {
                                            "box": [64, 48, 320, 240],
                                            "unit": "pixel",
                                        },
                                    },
                                ],
                            },
                        ]
                    },
                },
                {
                    "start": 2.0,
                    "end": 10.0,
                    "data": {
                        "frames": [
                            {
                                "timestamp": 20.0,
                                "detections": [
                                    {
                                        "label": "person",
                                        "score": 0.7,
                                        "box": {"box": [0, 0, 0.5, 0.5]},
                                    }
                                ],
                            }
                        ]
                    },
                },
            ]
        }
        model = _model(
            conn,
            analyzer="object_detection",
            model="rtdetr-v2-r50vd",
            sandbox_id="sb-1",
            sampling={"strategy": "interval", "every": 1},
        )

        frames = _predict(model, video, output)

        assert sorted(frames) == [31, 300]
        sign, car = frames[31].detections
        assert sign.label == "stop sign"
        assert sign.confidence == 0.5
        assert sign.bounding_box == [0.25, 0.5, 0.5, 0.5]
        assert car.bounding_box == pytest.approx([0.1, 0.1, 0.4, 0.4])
        (person,) = frames[300].detections
        assert person.bounding_box == [0, 0, 0.5, 0.5]

        kwargs = video.understand.call_args.kwargs
        assert kwargs["analyzers"] == [
            {
                "type": "object_detection",
                "name": "objects",
                "sampling": {"strategy": "interval", "every": 1},
                "config": {
                    "model": "rtdetr-v2-r50vd",
                    "sandbox_id": "sb-1",
                },
            }
        ]

    def test_object_detection_without_objects(self, conn, video):
        output = {
            "scenes": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "data": {"frames": [{"timestamp": 0.0, "detections": []}]},
                }
            ]
        }
        model = _model(conn, analyzer="object_detection")

        assert _predict(model, video, output) is None

    def test_default_analyzer_has_no_config(self, conn, video):
        model = _model(conn, analyzer="vlm")

        label = _predict(model, video, {"scenes": []})

        assert label.detections == []
        kwargs = video.understand.call_args.kwargs
        assert kwargs["analyzers"] == [{"type": "vlm", "name": "scene"}]

    def test_video_is_uploaded_once(self, conn, video):
        model = _model(conn)

        _predict(model, video, {"scenes": []})
        _predict(model, video, {"scenes": []})

        conn.get_collection.assert_called_once()
        conn.get_collection.return_value.upload.assert_called_once()
        assert video.understand.call_count == 2

    def test_upload_is_shared_across_models(self, conn, video):
        _predict(_model(conn), video, {"scenes": []})
        _predict(_model(conn, analyzer="ocr"), video, {"scenes": []})
        _predict(_model(conn, api_key="other-key"), video, {"scenes": []})
        _predict(_model(conn, collection_id="c-1"), video, {"scenes": []})

        # A different API key or collection needs its own upload
        assert conn.get_collection.return_value.upload.call_count == 3
        assert video.understand.call_count == 4

    def test_failed_analyzer_raises(self, conn, video):
        analyzer = _analyzer(status="failed")
        video.understand.return_value.get_analyzer.return_value = analyzer
        model = _model(conn)

        with mock.patch.object(
            model, "_get_video_metadata", return_value=_METADATA
        ):
            with pytest.raises(RuntimeError, match="status 'failed'"):
                model.predict(_VIDEO_URL)

        analyzer.get_output.assert_not_called()

    def test_unsupported_input_raises(self, conn):
        model = _model(conn)

        with pytest.raises(ValueError, match="Unsupported input"):
            model.predict(123)

    def test_video_metadata_from_reader(self):
        reader = mock.MagicMock(spec=fouv.etav.FFmpegVideoReader)
        reader.total_frame_count = 300
        reader.frame_rate = 30.0
        reader.frame_size = (640, 480)

        metadata = fouv.VideoDBModel._get_video_metadata(reader, "video.mp4")

        assert metadata.total_frame_count == 300
        assert metadata.frame_width == 640
        assert metadata.frame_height == 480
        assert metadata.duration == 10.0
