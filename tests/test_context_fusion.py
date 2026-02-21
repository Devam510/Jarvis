"""Tests for jarvis.cognition.context_fusion — Multi-modal fusion."""

from jarvis.cognition.context_fusion import ContextFusion, FusedContext


class TestReferenceDetection:
    def test_detect_this(self):
        cf = ContextFusion()
        refs = cf._detect_references("open this file")
        assert "this" in refs or "this_file" in refs

    def test_detect_that_error(self):
        cf = ContextFusion()
        refs = cf._detect_references("fix that error")
        assert "that_error" in refs

    def test_detect_here(self):
        cf = ContextFusion()
        refs = cf._detect_references("save it here")
        assert "here" in refs

    def test_no_references(self):
        cf = ContextFusion()
        refs = cf._detect_references("open chrome browser")
        assert len(refs) == 0


class TestFusion:
    def test_fuse_without_context(self):
        cf = ContextFusion()
        result = cf.fuse("open this file")
        assert result.original_transcript == "open this file"
        assert not result.has_visual_context

    def test_fuse_with_screen_context(self):
        cf = ContextFusion()
        ctx = {
            "app_name": "Visual Studio Code",
            "window_title": "main.py - Visual Studio Code",
            "ocr_text": "",
        }
        result = cf.fuse("open this file", ctx)
        assert result.has_visual_context
        assert result.screen_app == "Visual Studio Code"
        assert "main.py" in result.active_file

    def test_fuse_resolves_this_file(self):
        cf = ContextFusion()
        ctx = {
            "app_name": "Code",
            "window_title": "utils.py - Code",
            "ocr_text": "",
        }
        result = cf.fuse("refactor this file", ctx)
        assert (
            "this_file" in result.resolved_references
            or "this" in result.resolved_references
        )
        assert cf.total_references_resolved > 0

    def test_fuse_resolves_that_error(self):
        cf = ContextFusion()
        ctx = {
            "app_name": "Terminal",
            "window_title": "Terminal",
            "ocr_text": "TypeError: cannot add int and str",
        }
        result = cf.fuse("fix that error", ctx, recent_errors=None)
        if "that_error" in result.resolved_references:
            assert "TypeError" in result.resolved_references["that_error"]

    def test_fuse_resolves_this_app(self):
        cf = ContextFusion()
        ctx = {"app_name": "Chrome", "window_title": "Google", "ocr_text": ""}
        result = cf.fuse("close this app", ctx)
        if "this_app" in result.resolved_references:
            assert "Chrome" in result.resolved_references["this_app"]

    def test_fuse_with_recent_errors(self):
        cf = ContextFusion()
        ctx = {"app_name": "Code", "window_title": "test.py", "ocr_text": ""}
        errors = ["ImportError: no module named flask"]
        result = cf.fuse("fix that error", ctx, recent_errors=errors)
        if "that_error" in result.resolved_references:
            assert "ImportError" in result.resolved_references["that_error"]


class TestFileExtraction:
    def test_extract_from_vscode_title(self):
        file = ContextFusion._extract_active_file(
            "main.py - Visual Studio Code", "Code"
        )
        assert file == "main.py"

    def test_extract_from_path_title(self):
        file = ContextFusion._extract_active_file(
            "C:\\project\\utils.py - Code", "Code"
        )
        assert "utils.py" in file

    def test_extract_no_file(self):
        file = ContextFusion._extract_active_file("Google Chrome", "Chrome")
        assert file == ""

    def test_extract_empty(self):
        assert ContextFusion._extract_active_file("", "") == ""


class TestErrorDetection:
    def test_find_traceback(self):
        text = "File main.py, line 10\nTraceback most recent call last"
        error = ContextFusion._find_error_in_text(text)
        assert "Traceback" in error

    def test_find_type_error(self):
        text = "TypeError: unsupported operand"
        error = ContextFusion._find_error_in_text(text)
        assert "TypeError" in error

    def test_no_error(self):
        text = "Everything is fine, no problems here."
        error = ContextFusion._find_error_in_text(text)
        assert error == ""

    def test_empty_text(self):
        assert ContextFusion._find_error_in_text("") == ""


class TestBrowserPage:
    def test_resolve_this_page_chrome(self):
        cf = ContextFusion()
        ctx = {
            "app_name": "Google Chrome",
            "window_title": "Stack Overflow - Question",
            "ocr_text": "",
        }
        result = cf.fuse("summarize this page", ctx)
        if "this_page" in result.resolved_references:
            assert "Stack Overflow" in result.resolved_references["this_page"]


class TestStats:
    def test_fusion_counter(self):
        cf = ContextFusion()
        cf.fuse("hello")
        cf.fuse("world")
        assert cf.total_fusions == 2
