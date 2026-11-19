"""
`VideoDB <https://videodb.io>`_ video understanding integration.

This module provides a :class:`fiftyone.core.models.Model` that runs
`VideoDB <https://docs.videodb.io>`_ understanding analyzers on your videos
and stores their outputs as FiftyOne labels:

-   ``spoken_words``: timed transcript segments
-   ``vlm``: scene descriptions, optionally following your own prompt and
    output schema
-   ``ocr``: text visible on screen
-   ``brand_detection``: logos and brand names
-   ``activity_recognition``: actions and activities
-   ``location_detection``: setting and place
-   ``object_detection``: per-frame object bounding boxes

Object detections are stored as frame-level
:class:`fiftyone.core.labels.Detections`; all other analyzers are stored as
:class:`fiftyone.core.labels.TemporalDetections`.

The analyzers run server-side via the VideoDB API, so no local GPU is
required. Set your API key via the ``VIDEO_DB_API_KEY`` environment variable
or pass ``api_key=...`` when loading the model.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import logging
import os

import eta.core.video as etav

import fiftyone.core.config as foc
import fiftyone.core.labels as fol
import fiftyone.core.metadata as fome
import fiftyone.core.models as fom
import fiftyone.core.utils as fou
import fiftyone.zoo.models as fozm

videodb = fou.lazy_import(
    "videodb", callback=lambda: fou.ensure_package("videodb>=0.5.1")
)

logger = logging.getLogger(__name__)

# Maps each analyzer type to the name of the artifact it produces
_ANALYZERS = {
    "spoken_words": "transcript",
    "vlm": "scene",
    "ocr": "ocr",
    "brand_detection": "brands",
    "activity_recognition": "activity",
    "location_detection": "location",
    "object_detection": "objects",
}

# Analyzer types that accept each option
_OPTIONS = {
    "language": ("spoken_words",),
    "prompt": ("vlm",),
    "schema": ("vlm",),
    "inputs": ("vlm",),
    "sandbox_id": ("object_detection",),
    "model": tuple(t for t in _ANALYZERS if t != "spoken_words"),
    "sampling": tuple(t for t in _ANALYZERS if t != "spoken_words"),
}

# Videos uploaded in this session, keyed by (API key, collection ID, path), so
# that running several VideoDB models on a dataset uploads each video once
_UPLOADS = {}

# Label fields that analyzer outputs must not overwrite
_RESERVED_ATTRS = {
    "id",
    "_id",
    "_cls",
    "label",
    "support",
    "confidence",
    "tags",
    "attributes",
}


class VideoDBModelConfig(foc.Config, fozm.HasZooModel):
    """Configuration for running a :class:`VideoDBModel`.

    Args:
        analyzer ("spoken_words"): the VideoDB analyzer to run. Supported
            values are ``"spoken_words"``, ``"vlm"``, ``"ocr"``,
            ``"brand_detection"``, ``"activity_recognition"``,
            ``"location_detection"``, and ``"object_detection"``
        api_key (None): the VideoDB API key to use. If not provided, the
            ``VIDEO_DB_API_KEY`` environment variable is used
        collection_id (None): the ID of the VideoDB collection to upload
            videos to. By default, your default collection is used
        language (None): an optional language code, such as ``"en"``, for
            the ``"spoken_words"`` analyzer
        prompt (None): an optional prompt for the ``"vlm"`` analyzer. It can
            reference the outputs of ``inputs`` analyzers via
            ``{{inputs.<name>}}`` placeholders
        schema (None): an optional dict mapping field names to types, such
            as ``{"activity": "string"}``, that the ``"vlm"`` analyzer
            extracts for each scene
        inputs (None): an optional list of other analyzers whose outputs the
            ``"vlm"`` analyzer receives as context. Each entry is an analyzer
            type, such as ``"spoken_words"``, or a VideoDB analyzer dict,
            such as ``{"type": "ocr", "name": "ocr", "config": {...}}``.
            Only the ``"vlm"`` output is returned
        model (None): an optional model for the analyzer. For VLM-backed
            analyzers, this is a tier such as ``"mini"``, ``"basic"``,
            ``"pro"``, or ``"ultra"``. For ``"object_detection"``, this is a
            detection model such as ``"rtdetr-v2-r50vd"``
        sandbox_id (None): the ID of an active VideoDB sandbox to run a
            sandbox-backed ``"object_detection"`` model on
        sampling (None): an optional dict that controls which frames visual
            analyzers look at in each scene, such as
            ``{"strategy": "uniform", "frame_count": 4}`` or
            ``{"strategy": "interval", "every": 1}``
        segmentation (None): an optional dict that controls how videos are
            split into scenes, such as ``{"type": "time", "seconds": 5}`` or
            ``{"type": "shot", "threshold": 30}``
        transform (None): an optional dict of preprocessing to apply before
            analysis, such as ``{"resolution": "480p"}`` to reduce cost
        timeout (1800): the maximum time, in seconds, to wait for each video
            to be analyzed
        poll_interval (10): the time, in seconds, between status checks while
            waiting for each video to be analyzed
    """

    def __init__(self, d):
        self.analyzer = self.parse_string(
            d, "analyzer", default="spoken_words"
        )
        if self.analyzer not in _ANALYZERS:
            raise ValueError(
                "Unsupported analyzer '%s'. Supported values are %s"
                % (self.analyzer, tuple(_ANALYZERS.keys()))
            )

        self.api_key = self.parse_string(d, "api_key", default=None)
        self.collection_id = self.parse_string(
            d, "collection_id", default=None
        )
        self.language = self.parse_string(d, "language", default=None)
        self.prompt = self.parse_string(d, "prompt", default=None)
        self.schema = self.parse_dict(d, "schema", default=None)
        self.inputs = self.parse_array(d, "inputs", default=None)
        self.model = self.parse_string(d, "model", default=None)
        self.sandbox_id = self.parse_string(d, "sandbox_id", default=None)
        self.sampling = self.parse_dict(d, "sampling", default=None)
        self.segmentation = self.parse_dict(d, "segmentation", default=None)
        self.transform = self.parse_dict(d, "transform", default=None)
        self.timeout = self.parse_number(d, "timeout", default=1800)
        self.poll_interval = self.parse_number(d, "poll_interval", default=10)

        for name, analyzers in _OPTIONS.items():
            if (
                getattr(self, name) is not None
                and self.analyzer not in analyzers
            ):
                raise ValueError(
                    "%s is not supported for the '%s' analyzer"
                    % (name, self.analyzer)
                )

        # Validates the inputs up front so mistakes fail before any upload
        _parse_inputs(self.inputs)


class VideoDBModel(fom.Model):
    """Wrapper for running `VideoDB <https://videodb.io>`_ video understanding
    analyzers.

    Each video is uploaded to your VideoDB collection once per session, so
    running several VideoDB models on the same videos does not upload them
    again.
    ``"object_detection"`` returns frame-level
    :class:`fiftyone.core.labels.Detections`; every other analyzer returns
    :class:`fiftyone.core.labels.TemporalDetections` whose attributes hold the
    analyzer's output for each scene.

    Example usage::

        import fiftyone as fo
        import fiftyone.zoo as foz

        dataset = foz.load_zoo_dataset("quickstart-video", max_samples=2)

        # Timed transcript segments
        model = foz.load_zoo_model("videodb-transcript")
        dataset.apply_model(model, label_field="transcript")

        # Scene descriptions that also see the on-screen text
        model = foz.load_zoo_model(
            "videodb-scene-descriptions",
            prompt="Describe the scene. Visible text: {{inputs.ocr}}",
            inputs=["ocr"],
            segmentation={"type": "time", "seconds": 5},
        )
        dataset.apply_model(model, label_field="scenes")

        # Show the analyzer outputs as filterable fields in the App
        dataset.add_dynamic_sample_fields()

        session = fo.launch_app(dataset)

    Args:
        config: a :class:`VideoDBModelConfig`
    """

    def __init__(self, config):
        self.config = config

        api_key = config.api_key or os.environ.get("VIDEO_DB_API_KEY")
        if not api_key:
            raise ValueError(
                "No VideoDB API key found. Provide one via the `api_key` "
                "config parameter or the `VIDEO_DB_API_KEY` environment "
                "variable. You can get a free key at "
                "https://console.videodb.io"
            )

        self._api_key = api_key
        self._conn = videodb.connect(api_key=api_key)
        self._collection = None

    @property
    def media_type(self):
        return "video"

    @property
    def ragged_batches(self):
        return False

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self):
        return False

    @preprocess.setter
    def preprocess(self, value):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def predict(self, arg):
        """Analyzes the given video.

        Args:
            arg: an active ``eta.core.video.FFmpegVideoReader`` or a video
                filepath/URL

        Returns:
            a dict mapping frame numbers to
            :class:`fiftyone.core.labels.Detections` for
            ``"object_detection"``, or a
            :class:`fiftyone.core.labels.TemporalDetections` for all other
            analyzers
        """
        path = self._get_path(arg)
        video = self._get_video(path)
        output = self._understand(video)
        metadata = self._get_video_metadata(arg, path)

        if self.config.analyzer == "object_detection":
            return _parse_frames(output, metadata)

        return _parse_scenes(
            output, metadata, _ANALYZERS[self.config.analyzer]
        )

    @staticmethod
    def _get_path(arg):
        if isinstance(arg, etav.FFmpegVideoReader):
            return arg.inpath

        if isinstance(arg, str):
            return arg

        raise ValueError(
            "Unsupported input of type %s; expected a video filepath or an "
            "`eta.core.video.FFmpegVideoReader`" % type(arg)
        )

    @staticmethod
    def _get_video_metadata(arg, path):
        if isinstance(arg, etav.FFmpegVideoReader):
            total_frame_count = arg.total_frame_count
            frame_rate = arg.frame_rate
            width, height = arg.frame_size
            duration = total_frame_count / frame_rate if frame_rate else None
            return fome.VideoMetadata(
                total_frame_count=total_frame_count,
                frame_rate=frame_rate,
                frame_width=width,
                frame_height=height,
                duration=duration,
            )

        return fome.VideoMetadata.build_for(path)

    def _get_video(self, path):
        # Uploads each video once per session; upload() handles files and URLs
        collection_id = self.config.collection_id or "default"
        key = (self._api_key, collection_id, path)
        video = _UPLOADS.get(key, None)
        if video is not None:
            return video

        if self._collection is None:
            self._collection = self._conn.get_collection(collection_id)

        video = self._collection.upload(source=path)
        _UPLOADS[key] = video
        return video

    def _build_analyzers(self):
        analyzers = _parse_inputs(self.config.inputs)

        name = _ANALYZERS[self.config.analyzer]
        analyzer = {"type": self.config.analyzer, "name": name}
        if analyzers:
            analyzer["inputs"] = [a["name"] for a in analyzers]

        if self.config.sampling is not None:
            analyzer["sampling"] = self.config.sampling

        config = {}
        for key in ("language", "prompt", "schema", "model", "sandbox_id"):
            value = getattr(self.config, key)
            if value is not None:
                config[key] = value

        if config:
            analyzer["config"] = config

        analyzers.append(analyzer)
        return analyzers

    def _understand(self, video):
        name = _ANALYZERS[self.config.analyzer]

        understanding = video.understand(
            analyzers=self._build_analyzers(),
            segmentation=self.config.segmentation,
            transform=self.config.transform,
        )
        understanding.wait_until_complete(
            timeout=self.config.timeout,
            poll_interval=self.config.poll_interval,
        )

        analyzer = understanding.get_analyzer(name, refresh=True)
        if not analyzer.is_successful:
            raise RuntimeError(
                "VideoDB analyzer '%s' of understanding run '%s' finished "
                "with status '%s'" % (name, understanding.id, analyzer.status)
            )

        return analyzer.get_output()


def _parse_inputs(inputs):
    # VideoDB resolves `inputs` by analyzer name, so every input needs one
    analyzers = []
    for entry in inputs or []:
        analyzer = {"type": entry} if isinstance(entry, str) else dict(entry)
        if analyzer.get("type") not in _ANALYZERS:
            raise ValueError(
                "Unsupported input analyzer %r. Supported types are %s"
                % (entry, tuple(_ANALYZERS.keys()))
            )

        analyzer.setdefault("name", _ANALYZERS[analyzer["type"]])
        analyzers.append(analyzer)

    return analyzers


def _is_empty(value):
    return value is None or value == "" or value == [] or value == {}


def _parse_scenes(output, metadata, label):
    detections = []
    for scene in (output or {}).get("scenes") or []:
        data = scene.get("data") or {}

        # Skips scenes with no output, such as silent transcript segments
        if all(_is_empty(v) for v in data.values()):
            continue

        detection = fol.TemporalDetection.from_timestamps(
            [scene["start"], scene["end"]], metadata=metadata, label=label
        )

        for name, value in data.items():
            if name in _RESERVED_ATTRS:
                name += "_"

            detection[name] = value

        detections.append(detection)

    return fol.TemporalDetections(detections=detections)


def _parse_frames(output, metadata):
    frame_rate = metadata.frame_rate
    total_frame_count = metadata.total_frame_count

    frames = {}
    for scene in (output or {}).get("scenes") or []:
        for frame in (scene.get("data") or {}).get("frames") or []:
            frame_number = int(round(frame["timestamp"] * frame_rate)) + 1
            frame_number = min(max(frame_number, 1), total_frame_count)

            for det in frame.get("detections") or []:
                detection = _parse_detection(det, metadata)
                frames.setdefault(frame_number, fol.Detections())
                frames[frame_number].detections.append(detection)

    # An empty dict is not a valid frame labels dict, so no labels are added
    return frames or None


def _parse_detection(det, metadata):
    box = det["box"]
    x1, y1, x2, y2 = box["box"]

    if box.get("unit", "normalized") != "normalized":
        width, height = metadata.frame_width, metadata.frame_height
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height

    return fol.Detection(
        label=det["label"],
        bounding_box=[x1, y1, x2 - x1, y2 - y1],
        confidence=det.get("score"),
    )
