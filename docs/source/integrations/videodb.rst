.. _videodb-integration:

VideoDB Integration
===================

.. default-role:: code

FiftyOne integrates with `VideoDB <https://videodb.io>`_, whose video
understanding analyzers add timestamped labels to your video datasets with a
few lines of code:

.. list-table::
    :header-rows: 1

    * - Zoo model
      - VideoDB analyzer
      - Labels
    * - `videodb-transcript`
      - `spoken_words`
      - timed transcript segments with word-level timing and speakers
    * - `videodb-scene-descriptions`
      - `vlm`
      - scene descriptions, optionally following your own prompt and schema
    * - `videodb-ocr`
      - `ocr`
      - text visible on screen, such as signs, slides, and captions
    * - `videodb-brand-detection`
      - `brand_detection`
      - logos, brand names, and product labels
    * - `videodb-activity-recognition`
      - `activity_recognition`
      - the main action or activity in each scene
    * - `videodb-location-detection`
      - `location_detection`
      - the setting and place of each scene
    * - `videodb-object-detection`
      - `object_detection`
      - COCO object bounding boxes in sampled frames

Object detections are stored as frame-level
:class:`Detections <fiftyone.core.labels.Detections>`. All other analyzers are
stored as
:class:`TemporalDetections <fiftyone.core.labels.TemporalDetections>`, so you
can filter them, browse them on the App's timeline, and turn them into clips.
The analyzers run server-side via the VideoDB API, so no local GPU is
required.

.. _videodb-setup:

Setup
_____

Install the `videodb` package:

.. code-block:: shell
    :linenos:

    pip install "videodb>=0.5.1"

You can grab a free API key at
`console.videodb.io <https://console.videodb.io>`_. Provide it via the
``VIDEO_DB_API_KEY`` environment variable:

.. code-block:: shell
    :linenos:

    export VIDEO_DB_API_KEY=...

or pass it directly via the ``api_key`` config parameter when loading the
model.

.. note::

    Each video is uploaded to your VideoDB collection before it is analyzed.
    Uploads are reused for the rest of your Python session, so running
    several VideoDB models on the same videos uploads each video once. Pass
    ``collection_id=...`` to use a collection other than your default one.

.. _videodb-transcripts:

Video transcripts
_________________

Transcribe the speech in your videos into timed segments:

.. code-block:: python
    :linenos:

    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone.utils.videodb import VideoDBModel, VideoDBModelConfig

    dataset = foz.load_zoo_dataset("quickstart-video")

    # Load directly
    model = VideoDBModel(VideoDBModelConfig({"analyzer": "spoken_words"}))

    # Load via zoo
    # model = foz.load_zoo_model("videodb-transcript")

    dataset.apply_model(model, label_field="transcript")

    # Show the `text` and `words` attributes in the App's sidebar
    dataset.add_dynamic_sample_fields()

    session = fo.launch_app(dataset)

Each transcript segment is a
:class:`TemporalDetection <fiftyone.core.labels.TemporalDetection>` with label
``"transcript"``, whose ``text`` attribute holds the spoken text and whose
``words`` attribute holds the timing, confidence, and speaker of each word.
Segments without speech are skipped, so silent videos have no segments.

You can pass a language hint for non-English videos:

.. code-block:: python
    :linenos:

    model = foz.load_zoo_model("videodb-transcript", language="hi")

.. note::

    Analyzer outputs are stored as label attributes. Call
    :meth:`add_dynamic_sample_fields() <fiftyone.core.dataset.Dataset.add_dynamic_sample_fields>`
    after applying a model so the App lists them in its sidebar, where you
    can filter by them.

.. _videodb-scenes:

Scene descriptions
__________________

Split your videos into scenes and describe each one:

.. code-block:: python
    :linenos:

    model = foz.load_zoo_model("videodb-scene-descriptions")

    dataset.apply_model(model, label_field="scenes")
    dataset.add_dynamic_sample_fields()

Each scene is a
:class:`TemporalDetection <fiftyone.core.labels.TemporalDetection>` with label
``"scene"`` whose ``text`` attribute holds its description.

By default, scenes are split at visual cuts. You can split them into fixed
windows instead, and provide your own prompt and an output ``schema`` whose
fields are stored as attributes of each scene:

.. code-block:: python
    :linenos:

    from fiftyone import ViewField as F

    model = foz.load_zoo_model(
        "videodb-scene-descriptions",
        prompt="Describe the activity and the setting of this scene.",
        schema={"activity": "string", "setting": "string"},
        segmentation={"type": "time", "seconds": 5},
        model="pro",  # "mini", "basic", "pro", or "ultra"
    )

    dataset.apply_model(model, label_field="activities")
    dataset.add_dynamic_sample_fields()

    # Find all outdoor scenes
    view = dataset.filter_labels(
        "activities", F("setting").contains_str("outdoor")
    )

    # Turn each matching scene into its own clip
    clips = view.to_clips("activities")

    session = fo.launch_app(clips)

.. _videodb-combining:

Combining analyzers
-------------------

Scene descriptions can use the outputs of other analyzers as context. List
them in ``inputs`` and reference them in your prompt with
``{{inputs.<name>}}`` placeholders, where ``<name>`` is the analyzer's
artifact name, such as ``transcript`` or ``ocr``:

.. code-block:: python
    :linenos:

    model = foz.load_zoo_model(
        "videodb-scene-descriptions",
        inputs=["spoken_words", "ocr"],
        prompt=(
            "Describe this scene. What was said: {{inputs.transcript}} "
            "Text on screen: {{inputs.ocr}}"
        ),
    )

    dataset.apply_model(model, label_field="scenes")

Each entry of ``inputs`` is an analyzer type, or a VideoDB analyzer dict such
as ``{"type": "ocr", "name": "signs", "config": {"model": "pro"}}`` when you
need a custom name or configuration. Only the scene descriptions are stored.

.. _videodb-visual-signals:

On-screen text, brands, activities, and locations
_________________________________________________

Four specialized analyzers extract one kind of signal from each scene:

.. code-block:: python
    :linenos:

    from fiftyone import ViewField as F

    windows = {"type": "time", "seconds": 5}

    dataset.apply_model(
        foz.load_zoo_model("videodb-ocr", segmentation=windows),
        label_field="ocr",
    )
    dataset.apply_model(
        foz.load_zoo_model("videodb-brand-detection", segmentation=windows),
        label_field="brands",
    )
    dataset.apply_model(
        foz.load_zoo_model(
            "videodb-activity-recognition", segmentation=windows
        ),
        label_field="activity",
    )
    dataset.apply_model(
        foz.load_zoo_model("videodb-location-detection", segmentation=windows),
        label_field="location",
    )
    dataset.add_dynamic_sample_fields()

    # Find the moments where a given brand is visible
    view = dataset.filter_labels(
        "brands", F("brand_names").contains("Nike")
    )
    clips = view.to_clips("brands")

Each scene's output is stored on its
:class:`TemporalDetection <fiftyone.core.labels.TemporalDetection>`:

.. list-table::
    :header-rows: 1

    * - Zoo model
      - Label
      - Attributes
    * - `videodb-ocr`
      - ``"ocr"``
      - ``text``
    * - `videodb-brand-detection`
      - ``"brands"``
      - ``brand_names`` and ``mentions``, the reason each brand was found
    * - `videodb-activity-recognition`
      - ``"activity"``
      - ``activity``
    * - `videodb-location-detection`
      - ``"location"``
      - ``location``

Each of these models also accepts a ``model`` tier.

.. _videodb-objects:

Object detection
________________

Detect objects with bounding boxes in sampled frames of your videos:

.. code-block:: python
    :linenos:

    model = foz.load_zoo_model(
        "videodb-object-detection",
        sandbox_id="...",
        sampling={"strategy": "interval", "every": 1},
    )

    dataset.apply_model(model, label_field="objects")

    session = fo.launch_app(dataset)

Detections are stored in the frame-level ``objects`` field of each sampled
frame, using the COCO label vocabulary.

.. note::

    The default object detection model runs on a
    `VideoDB sandbox <https://docs.videodb.io/pages/core-concepts/sandbox-compute>`_.
    Start one with ``model_categories=["object_detection"]`` and pass its ID
    via ``sandbox_id``.

.. _videodb-cost:

Controlling cost and latency
____________________________

These options apply to every VideoDB model:

.. list-table::
    :header-rows: 1

    * - Option
      - Description
    * - ``segmentation``
      - How videos are split into scenes, such as
        ``{"type": "time", "seconds": 10}`` or
        ``{"type": "shot", "threshold": 30}``. Longer scenes mean fewer
        analyzer calls
    * - ``transform``
      - Preprocessing applied before analysis, such as
        ``{"resolution": "480p"}``
    * - ``sampling``
      - The frames visual analyzers look at in each scene, such as
        ``{"strategy": "uniform", "frame_count": 4}`` or
        ``{"strategy": "interval", "every": 1}``. Not supported for
        transcripts
    * - ``model``
      - The model tier of VLM-backed analyzers: ``"mini"``, ``"basic"``,
        ``"pro"``, or ``"ultra"``

.. code-block:: python
    :linenos:

    model = foz.load_zoo_model(
        "videodb-scene-descriptions",
        segmentation={"type": "time", "seconds": 30},
        transform={"resolution": "480p"},
        sampling={"strategy": "uniform", "frame_count": 1},
        model="mini",
    )
