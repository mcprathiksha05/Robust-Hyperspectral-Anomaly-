from src.analysis.research_assistant import answer_research_question


def test_research_assistant_uses_active_dataset_facts():
    answer = answer_research_question(
        "What is the active dataset?",
        "scene-a",
        (10, 20, 30),
        5,
        (4, 8),
        None,
        "Global-RX",
        None,
        None,
        [],
    )
    assert "scene-a" in answer
    assert "(10, 20, 30)" in answer
    assert "5 positive reference pixels" in answer


def test_research_assistant_does_not_invent_a_pixel_score():
    answer = answer_research_question(
        "Explain this pixel score",
        "scene-a",
        (10, 20, 30),
        5,
        (4, 8),
        None,
        "Global-RX",
        None,
        None,
        [],
    )
    assert "has not been scored yet" in answer
    assert "not available" not in answer


def test_research_assistant_reports_measured_memory_only():
    runs = [
        {
            "status": "success",
            "detector_name": "PCA-Residual",
            "experiment_id": "run-1",
            "memory": {"peak_traced_mb": 2.5},
        },
        {"status": "success", "detector_name": "Global-RX", "memory": None},
    ]
    answer = answer_research_question(
        "Which model used the least memory?",
        "scene-a",
        (10, 20, 30),
        5,
        (4, 8),
        None,
        "Global-RX",
        None,
        None,
        runs,
    )
    assert "2.500 MiB" in answer
    assert "run-1" in answer


def test_research_assistant_declines_unsupported_claims():
    answer = answer_research_question(
        "Will this detector identify a target?",
        "scene-a",
        (10, 20, 30),
        5,
        (4, 8),
        None,
        "Global-RX",
        None,
        None,
        [],
    )
    assert "cannot support that question" in answer
