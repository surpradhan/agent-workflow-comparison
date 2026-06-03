"""Tests for task registry and definitions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks import TaskLevel, registry


class TestTaskRegistry:
    def test_all_tasks_loaded(self):
        tasks = registry.all()
        assert len(tasks) == 25  # 8 + 7 + 5 + 5 (original 18 + 7 added for stat power)

    def test_level_distribution(self):
        assert len(registry.get_by_level(TaskLevel.RETRIEVAL)) == 8
        assert len(registry.get_by_level(TaskLevel.ANALYTICAL)) == 7
        assert len(registry.get_by_level(TaskLevel.REASONING)) == 5
        assert len(registry.get_by_level(TaskLevel.DECISION)) == 5

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
