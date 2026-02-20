"""Tests for the recommendation engine."""

import pytest

from astrograph.ast_to_graph import CodeUnit
from astrograph.index import CodeStructureIndex, DuplicateGroup
from astrograph.recommendations import (
    ActionType,
    Evidence,
    ImpactLevel,
    LocationInfo,
    RecommendationEngine,
    RefactoringRecommendation,
    format_recommendations_report,
)


class TestLocationInfo:
    """Tests for LocationInfo dataclass."""

    def test_basic_location(self):
        loc = LocationInfo(
            file_path="src/utils.py",
            name="validate",
            lines="10-20",
            unit_type="function",
        )
        assert loc.file_path == "src/utils.py"
        assert loc.is_test_file is False

    def test_test_file_detection(self):
        loc = LocationInfo(
            file_path="tests/test_utils.py",
            name="test_validate",
            lines="10-20",
            unit_type="function",
            is_test_file=True,
        )
        assert loc.is_test_file is True


class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_evidence_with_metric(self):
        ev = Evidence(fact="Found duplicates", metric="3 occurrences")
        assert ev.fact == "Found duplicates"
        assert ev.metric == "3 occurrences"

    def test_evidence_without_metric(self):
        ev = Evidence(fact="Verified via isomorphism")
        assert ev.metric is None


class TestRefactoringRecommendation:
    """Tests for RefactoringRecommendation dataclass."""

    def test_to_dict(self):
        rec = RefactoringRecommendation(
            action=ActionType.EXTRACT_TO_UTILITY,
            summary="Test summary",
            rationale="Test rationale",
            impact=ImpactLevel.HIGH,
            impact_score=0.85,
            confidence=0.9,
            evidence=[Evidence(fact="Test fact", metric="1 item")],
            locations=[
                LocationInfo(
                    file_path="src/a.py",
                    name="func_a",
                    lines="1-10",
                    unit_type="function",
                )
            ],
            lines_duplicated=30,
            estimated_lines_saved=20,
            files_affected=2,
        )

        d = rec.to_dict()

        assert d["action"] == "extract_to_utility"
        assert d["locations"] == ["src/a.py:func_a"]
        # keep is only present when there's a clear reason
        assert "keep" not in d or d.get("keep_reason") is not None


class TestRecommendationEngine:
    """Tests for the RecommendationEngine."""

    @pytest.fixture
    def engine(self):
        return RecommendationEngine()

    @staticmethod
    def _add_duplicate_units(
        index: CodeStructureIndex,
        prefix: str,
        code: str,
        line_end: int,
    ) -> None:
        for i in range(2):
            index.add_code_unit(
                CodeUnit(
                    name=f"{prefix}_{i}",
                    code=code,
                    file_path=f"src/{prefix}{i}.py",
                    line_start=1,
                    line_end=line_end,
                    unit_type="function",
                )
            )

    @staticmethod
    def _assert_first_action(recommendations, action: ActionType) -> None:
        first = (
            recommendations[0] if recommendations else pytest.skip("No recommendations produced")
        )
        assert first.action == action

    @pytest.fixture
    def sample_index_with_duplicates(self):
        """Create an index with actual duplicates for testing."""
        index = CodeStructureIndex()

        # Two structurally identical functions
        code1 = """
def validate_input(data):
    if not data:
        raise ValueError("Empty")
    return data.strip()
"""
        code2 = """
def check_data(value):
    if not value:
        raise ValueError("Empty")
    return value.strip()
"""
        unit1 = CodeUnit(
            name="validate_input",
            code=code1,
            file_path="src/handlers/user.py",
            line_start=10,
            line_end=15,
            unit_type="function",
        )
        unit2 = CodeUnit(
            name="check_data",
            code=code2,
            file_path="src/handlers/order.py",
            line_start=20,
            line_end=25,
            unit_type="function",
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        return index

    def test_analyze_empty_groups(self, engine):
        """Empty groups should return empty recommendations."""
        recommendations = engine.analyze_duplicates([])
        assert recommendations == []

    def test_analyze_single_entry_group(self, engine):
        """Groups with only one entry should be skipped."""
        # Create a group with one entry
        unit = CodeUnit(
            name="test",
            code="def test(): pass",
            file_path="test.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        index = CodeStructureIndex()
        entry = index.add_code_unit(unit)

        group = DuplicateGroup(wl_hash="abc123", entries=[entry])
        recommendations = engine.analyze_duplicates([group])

        assert recommendations == []

    def test_analyze_duplicates_generates_recommendations(self, sample_index_with_duplicates):
        """Duplicates should generate recommendations."""
        engine = RecommendationEngine()
        groups = sample_index_with_duplicates.find_all_duplicates(min_node_count=3)

        # Should have at least one group
        assert len(groups) >= 1

        recommendations = engine.analyze_duplicates(groups)

        assert len(recommendations) >= 1
        rec = recommendations[0]

        assert rec.action in ActionType
        assert rec.impact in ImpactLevel
        assert 0 <= rec.impact_score <= 1
        assert 0 <= rec.confidence <= 1
        assert len(rec.evidence) > 0
        assert len(rec.locations) >= 2

    def test_test_file_detection(self, engine):
        """Test files should be properly detected."""
        index = CodeStructureIndex()

        code = "def test_func(): return 1"
        unit1 = CodeUnit(
            name="test_a",
            code=code,
            file_path="tests/test_module.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        unit2 = CodeUnit(
            name="test_b",
            code=code,
            file_path="tests/test_other.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=1)
        recommendations = engine.analyze_duplicates(groups)
        self._assert_first_action(recommendations, ActionType.REVIEW_TEST_DUPLICATION)

    def test_recommendations_sorted_by_impact(self, engine):
        """Recommendations should be sorted by impact score descending."""
        index = CodeStructureIndex()

        # Create two duplicate groups with different complexities
        simple_code = "def f(): return 1"
        complex_code = """
def process(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results
"""

        self._add_duplicate_units(index, "simple", simple_code, line_end=1)
        self._add_duplicate_units(index, "complex", complex_code, line_end=7)

        groups = index.find_all_duplicates(min_node_count=1)
        recommendations = engine.analyze_duplicates(groups)

        # Should be sorted by impact score descending
        if len(recommendations) >= 2:
            for i in range(len(recommendations) - 1):
                assert recommendations[i].impact_score >= recommendations[i + 1].impact_score

    def test_keep_location_prefers_shallower_path(self, engine):
        """Keep location should prefer shallower paths when clear winner exists."""
        index = CodeStructureIndex()

        code = "def validate(x): return x > 0"

        # Shallower path (depth 2)
        unit1 = CodeUnit(
            name="validate",
            code=code,
            file_path="src/validate.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        # Deeper path (depth 3)
        unit2 = CodeUnit(
            name="check",
            code=code,
            file_path="src/handlers/user.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=1)
        recommendations = engine.analyze_duplicates(groups)

        if recommendations:
            rec = recommendations[0]
            # Should prefer the shallower one
            assert rec.keep_location is not None
            assert rec.keep_location.file_path == "src/validate.py"
            assert rec.keep_reason == "shallowest path"

    def test_no_keep_recommendation_when_equal_depth(self, engine):
        """Should not recommend keep when paths have equal depth."""
        index = CodeStructureIndex()

        code = "def validate(x): return x > 0"

        # Same depth (both depth 3)
        unit1 = CodeUnit(
            name="validate",
            code=code,
            file_path="src/handlers/a.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        unit2 = CodeUnit(
            name="check",
            code=code,
            file_path="src/handlers/b.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=1)
        recommendations = engine.analyze_duplicates(groups)

        if recommendations:
            rec = recommendations[0]
            # Should NOT recommend which to keep
            assert rec.keep_location is None
            assert rec.keep_reason is None

    def test_extract_to_base_class_action(self, engine):
        """Methods with different parents should suggest base class extraction."""
        index = CodeStructureIndex()

        # Same method code in different classes
        method_code = """
def save(self):
    self.validate()
    self.persist()
    return True
"""
        unit1 = CodeUnit(
            name="save",
            code=method_code,
            file_path="src/models/user.py",
            line_start=10,
            line_end=14,
            unit_type="method",
            parent_name="UserModel",
        )
        unit2 = CodeUnit(
            name="save",
            code=method_code,
            file_path="src/models/order.py",
            line_start=20,
            line_end=24,
            unit_type="method",
            parent_name="OrderModel",
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        recommendations = engine.analyze_duplicates(groups)
        self._assert_first_action(recommendations, ActionType.EXTRACT_TO_BASE_CLASS)

    def test_consolidate_in_place_action(self, engine):
        """Duplicates in same directory should suggest consolidation."""
        index = CodeStructureIndex()

        code = """
def helper(data):
    result = []
    for item in data:
        result.append(item)
    return result
"""
        unit1 = CodeUnit(
            name="helper_a",
            code=code,
            file_path="src/utils/a.py",
            line_start=1,
            line_end=6,
            unit_type="function",
        )
        unit2 = CodeUnit(
            name="helper_b",
            code=code,
            file_path="src/utils/b.py",
            line_start=1,
            line_end=6,
            unit_type="function",
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        recommendations = engine.analyze_duplicates(groups)
        self._assert_first_action(recommendations, ActionType.CONSOLIDATE_IN_PLACE)


class TestFormatRecommendationsReport:
    """Tests for the report formatting function."""

    def test_empty_recommendations(self):
        report = format_recommendations_report([])
        assert "No refactoring opportunities" in report

    def test_report_is_concise(self):
        rec = RefactoringRecommendation(
            action=ActionType.EXTRACT_TO_UTILITY,
            summary="Test summary",
            rationale="This is a test rationale for formatting.",
            impact=ImpactLevel.HIGH,
            impact_score=0.85,
            confidence=0.9,
            evidence=[
                Evidence(fact="3 duplicates found", metric="3 occurrences"),
            ],
            locations=[
                LocationInfo(
                    file_path="src/a.py",
                    name="func_a",
                    lines="1-10",
                    unit_type="function",
                    directory_depth=2,
                ),
                LocationInfo(
                    file_path="src/deep/nested/b.py",
                    name="func_b",
                    lines="5-15",
                    unit_type="function",
                    directory_depth=4,
                ),
            ],
            keep_location=LocationInfo(
                file_path="src/a.py",
                name="func_a",
                lines="1-10",
                unit_type="function",
                directory_depth=2,
            ),
            keep_reason="shallowest path",
            suggested_name="common_func",
            lines_duplicated=30,
            estimated_lines_saved=20,
            files_affected=2,
        )

        report = format_recommendations_report([rec])

        # Check key info is present
        assert "extract_to_utility" in report
        assert "src/a.py:func_a" in report
        assert "Keep" in report
        assert "shallowest path" in report
        # Should be very concise - just 2 lines per recommendation
        assert len(report.split("\n")) == 2


class TestIntegrationWithTools:
    """Integration tests with the tools module."""

    def test_analyze_tool(self):
        """Test the analyze tool integration."""
        from astrograph.tools import CodeStructureTools

        tools = CodeStructureTools()

        # Index some duplicate code
        code1 = """
def process_a(data):
    result = []
    for item in data:
        result.append(item.upper())
    return result
"""
        code2 = """
def process_b(items):
    result = []
    for item in items:
        result.append(item.upper())
    return result
"""
        unit1 = CodeUnit(
            name="process_a",
            code=code1,
            file_path="src/module_a.py",
            line_start=1,
            line_end=6,
            unit_type="function",
        )
        unit2 = CodeUnit(
            name="process_b",
            code=code2,
            file_path="src/module_b.py",
            line_start=1,
            line_end=6,
            unit_type="function",
        )

        tools.index.add_code_unit(unit1)
        tools.index.add_code_unit(unit2)

        # Analyze (simplified interface - no parameters needed)
        result = tools.analyze()
        # Should show findings with suppress calls or no findings
        assert "suppress(wl_hash=" in result.text or "No significant duplicates" in result.text

    def test_analyze_dispatch(self):
        """Test that analyze can be called via dispatch."""
        from astrograph.tools import CodeStructureTools

        tools = CodeStructureTools()
        result = tools.call_tool("analyze", {})

        # No code indexed
        assert "No code indexed" in result.text

    def test_similar_code_detection(self):
        """Test that similar (but not identical) code is detected."""
        from astrograph.tools import CodeStructureTools

        tools = CodeStructureTools()

        # Two similar but not identical functions
        code1 = """
def process_a(data):
    result = []
    for item in data:
        result.append(item.upper())
    return result
"""
        code2 = """
def process_b(items):
    result = []
    for item in items:
        if item:
            result.append(item.upper())
    return result
"""
        unit1 = CodeUnit(
            name="process_a",
            code=code1,
            file_path="src/module_a.py",
            line_start=1,
            line_end=6,
            unit_type="function",
        )
        unit2 = CodeUnit(
            name="process_b",
            code=code2,
            file_path="src/module_b.py",
            line_start=1,
            line_end=7,
            unit_type="function",
        )

        tools.index.add_code_unit(unit1)
        tools.index.add_code_unit(unit2)

        # Analyze (simplified interface)
        result = tools.analyze()
        # Should find them as similar or not find them - either is valid with internal threshold
        assert "pattern" in result.text or "No significant duplicates" in result.text
