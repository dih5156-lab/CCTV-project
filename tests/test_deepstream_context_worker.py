from queue import Queue
from unittest.mock import MagicMock

from src.core._deepstream_context_worker import DeepStreamContextWorker, FaceContextTask
from src.core.events import DetectionEvent, EventType


def _person_event() -> DetectionEvent:
    return DetectionEvent(
        EventType.PERSON,
        1,
        2,
        30,
        40,
        0.9,
        1.0,
        object_id=7,
    )


def test_context_worker_submit_queues_person_task() -> None:
    queue = Queue(maxsize=2)
    person = _person_event()
    worker = DeepStreamContextWorker(
        queue=queue,
        feature_flags_for_camera=lambda camera_name: {
            "use_face": True,
            "use_appearance": False,
        },
        remember_context_events=MagicMock(),
        collect_context_events=lambda camera_name, events: list(events),
        get_camera_frame=lambda camera_name: object(),
        run_face_recognition=MagicMock(return_value=[]),
        log_appearance_capability_hints=MagicMock(),
        refresh_appearance_conditions=MagicMock(),
        appearance_pipeline=MagicMock(),
        enqueue_event=MagicMock(return_value=True),
    )

    worker.submit("cam1", [person])

    task = queue.get_nowait()
    assert task.camera_name == "cam1"
    assert task.person_events == [person]
    assert task.flags["use_face"] is True


def test_context_worker_process_adds_appearance_metadata() -> None:
    person = _person_event()
    appearance_event = DetectionEvent(
        EventType.PERSON,
        3,
        4,
        20,
        10,
        0.8,
        1.0,
        object_id=7,
    )
    appearance_pipeline = MagicMock()
    appearance_pipeline.run.return_value = [appearance_event]
    enqueue_event = MagicMock(return_value=True)
    worker = DeepStreamContextWorker(
        queue=Queue(),
        feature_flags_for_camera=MagicMock(),
        remember_context_events=MagicMock(),
        collect_context_events=MagicMock(),
        get_camera_frame=MagicMock(),
        run_face_recognition=MagicMock(return_value=[]),
        log_appearance_capability_hints=MagicMock(),
        refresh_appearance_conditions=MagicMock(),
        appearance_pipeline=appearance_pipeline,
        enqueue_event=enqueue_event,
    )

    worker.process(
        FaceContextTask(
            camera_name="cam1",
            person_events=[person],
            frame=object(),
            flags={"use_face": False, "use_appearance": True},
            context_events=[person],
        )
    )

    assert appearance_event.metadata["backend"] == "deepstream_context"
    assert appearance_event.metadata["camera_id"] == "cam1"
    enqueue_event.assert_called_once_with(appearance_event, "cam1")
