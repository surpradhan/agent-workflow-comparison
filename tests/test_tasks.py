"""Tests for task registry and definitions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks import registry, TaskLevel


class TestTaskRegistry:
    def test_all_tasks_loaded(self):
        tasks = registry.all()
        assert len(tasks) == 18  # 5 + 5 + 4 + 4

    def test_level_distribution(self):
        assert len(registry.get_by_level(TaskLevel.RETRIEVAL)) == 5
        assert len(registry.get_by_level(TaskLevel.ANALYTICAL)) == 5
        assert len(registry.get_by_level(TaskLevel.REASONING)) == 4
        assert len(registry.get_by_level(TaskLevel.DECISION)) == 4

    def test_get_task_by_id(self):
        task = registry.get("L1_T01")
        assert task.level == TaskLevel.RETRIEVAL
        assert "revenue" in task.description.lower()

    def test_all_tasks_have_expected_tools(self):
        for task in registry.all():
            assert len(task.expected_tools) > 0, f"Task {task.id} has no expected tools"

    def test_all_tasks_have_evaluation_criteria(self):
        for task in registry.all():
            assert task.evaluation_criteria, f"Task {task.id} has no evaluation criteria"
