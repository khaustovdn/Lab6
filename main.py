#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import time
import logging
import traceback
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Callable,
    DefaultDict,
)

from PySide6.QtCore import QObject, Signal, Slot, QSettings
from PySide6.QtGui import QAction, QColor, QTextCharFormat, QTextCursor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# region Core Domain Types


class LogLevel(Enum):
    """Enumeration of log severity levels."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass(frozen=True)
class TextPosition:
    """Immutable text position representation."""
    line: int
    column: int
    absolute: int


class MatchType(Enum):
    """Pattern match categories."""
    TIME = auto()
    BITCOIN = auto()
    COMMENT = auto()


@dataclass(frozen=True)
class PatternMatch:
    """Domain model for text pattern matches."""
    text: str
    kind: MatchType
    start: TextPosition
    end: TextPosition


@dataclass(frozen=True)
class AnalysisMetrics:
    """Performance and statistical metrics."""
    total_matches: int
    processing_time: float


@dataclass(frozen=True)
class AnalysisResult:
    """Result container for text analysis."""
    matches: List[PatternMatch]
    metrics: AnalysisMetrics

# endregion

# region Exceptions


class FileOperationError(Exception):
    """Exception raised for file operation failures."""

    def __init__(self, message: str, original: Optional[Exception] = None):
        super().__init__(message)
        self.original = original

# endregion

# region Application Interfaces


@runtime_checkable
class ILogger(Protocol):
    """Abstract logging interface."""
    @abstractmethod
    def log(self, level: LogLevel, message: str, **metadata: Any) -> None:
        ...


@runtime_checkable
class IFileHandler(Protocol):
    """Abstract file operations."""
    @abstractmethod
    def read(self, path: Path) -> str:
        ...

    @abstractmethod
    def write(self, path: Path, content: str) -> None:
        ...


@runtime_checkable
class ITextAnalyzer(Protocol):
    """Text analysis contract."""
    @abstractmethod
    def analyze(self, text: str) -> AnalysisResult:
        ...


@runtime_checkable
class ISettingsManager(Protocol):
    """Configuration management interface."""
    @abstractmethod
    def get(self, section: str, key: str, default: Any = None) -> Any:
        ...

    @abstractmethod
    def set(self, section: str, key: str, value: Any) -> None:
        ...

    @abstractmethod
    def register_handler(self, section: str, handler: Callable[[str, Any], None]) -> None:
        ...


class IView(Protocol):
    """Abstract view interface."""
    @abstractmethod
    def display_results(self, result: AnalysisResult) -> None:
        ...

    @abstractmethod
    def set_content(self, text: str) -> None:
        ...

    @abstractmethod
    def show(self) -> None:
        ...

    @property
    @abstractmethod
    def content_changed(self) -> Signal:
        ...

    @property
    @abstractmethod
    def file_open_requested(self) -> Signal:
        ...

# endregion


# region Theme Abstraction

class ITheme(Protocol):
    """Theme interface following Open/Closed principle"""
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def colors(self) -> Dict[str, QColor]:
        ...

    @abstractmethod
    def apply(self, app: QApplication) -> None:
        ...

    @abstractmethod
    def get_stylesheet(self) -> str:
        ...


class ThemeManager(QObject):
    """Centralized theme management (Single Responsibility)"""
    theme_changed = Signal(ITheme)

    def __init__(self, settings: ISettingsManager):
        super().__init__()
        self._settings = settings
        self._themes: Dict[str, ITheme] = {}
        self._current_theme: Optional[ITheme] = None

    def register_theme(self, theme: ITheme) -> None:
        """Open for extension through registration"""
        self._themes[theme.name] = theme

    def set_theme(self, theme_name: str) -> None:
        """Set current theme using strategy pattern"""
        if theme := self._themes.get(theme_name):
            self._current_theme = theme
            self._settings.set("ui", "theme", theme_name)
            self.theme_changed.emit(theme)

    def current_theme(self) -> ITheme:
        """Liskov Substitution: Return any registered ITheme"""
        return self._current_theme or next(iter(self._themes.values()))


class CatppuccinDarkTheme(ITheme):
    """Concrete Catppuccin theme implementation"""

    def __init__(self):
        self._colors = {
            "base": QColor("#1E1E2E"),
            "mantle": QColor("#181825"),
            "text": QColor("#CDD6F4"),
            "subtext": QColor("#BAC2DE"),
            "blue": QColor("#89B4FA"),
            "red": QColor("#F38BA8"),
            "green": QColor("#A6E3A1"),
            "yellow": QColor("#F9E2AF"),
            "accent": QColor("#CBA6F7"),
        }

        self._stylesheet = f"""
            QWidget {{
                background-color: {self._colors["base"].name()};
                color: {self._colors["text"].name()};
                font-family: "Fira Code";
            }}

            QPlainTextEdit {{
                background-color: {self._colors["mantle"].name()};
                border: 1px solid {self._colors["accent"].name()};
            }}

            QTableWidget::item {{
                selection-background-color: {self._colors["blue"].name()};
            }}

            /* Add more styled components here */
        """

    @property
    def name(self) -> str:
        return "catppuccin-dark"

    @property
    def colors(self) -> Dict[str, QColor]:
        return self._colors

    def apply(self, app: QApplication) -> None:
        app.setStyle("Fusion")
        app.setPalette(self._create_palette())
        app.setStyleSheet(self._stylesheet)

    def get_stylesheet(self) -> str:
        return self._stylesheet

    def _create_palette(self) -> QPalette:
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, self._colors["base"])
        palette.setColor(QPalette.ColorRole.WindowText, self._colors["text"])
        palette.setColor(QPalette.ColorRole.Base, self._colors["mantle"])
        return palette

# endregion

# region Infrastructure


class StructuredLogger(ILogger):
    """Production-grade structured logger."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("TextAnalyzerPro")
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Initialize logging infrastructure."""
        self._logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s || %(metadata)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

    def log(self, level: LogLevel, message: str, **metadata: Any) -> None:
        """Log structured message with metadata."""
        log_level = self._convert_level(level)
        self._logger.log(log_level, message, extra={"metadata": metadata or {}})

    def _convert_level(self, level: LogLevel) -> int:
        """Convert custom log level to standard."""
        return {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }[level]


class SecureFileHandler(IFileHandler):
    """Secure file operations with validation."""

    VALID_EXTENSIONS = {".txt", ".log", ".csv"}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def read(self, path: Path) -> str:
        """Safely read text file with validation."""
        self._validate_path(path)
        try:
            content = path.read_text(encoding="utf-8", errors="strict")
            self._logger.log(
                LogLevel.INFO,
                "File read successful",
                path=str(path),
                size=len(content)
            )
            return content
        except Exception as e:
            self._logger.log(
                LogLevel.ERROR,
                "File read failure",
                error=str(e),
                traceback=traceback.format_exc())
            raise FileOperationError(f"Read failed: {e}", e) from e

    def write(self, path: Path, content: str) -> None:
        """Securely write content to file."""
        self._validate_path(path)
        try:
            path.write_text(content, encoding="utf-8", errors="strict")
            self._logger.log(
                LogLevel.INFO,
                "File write successful",
                path=str(path),
                size=len(content))
        except Exception as e:
            self._logger.log(
                LogLevel.ERROR,
                "File write failure",
                error=str(e),
                traceback=traceback.format_exc())
            raise FileOperationError(f"Write failed: {e}", e) from e

    def _validate_path(self, path: Path) -> None:
        """Perform security and validation checks."""
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        if path.suffix not in self.VALID_EXTENSIONS:
            raise ValueError(f"Invalid file extension: {path.suffix}")
        if path.stat().st_size > self.MAX_FILE_SIZE:
            raise ValueError("File size exceeds maximum allowed")

# endregion

# region Business Logic


class RegexAnalyzer(ITextAnalyzer):
    """Configurable regex-based text analyzer."""

    DEFAULT_PATTERNS = {
        "TIME": r"\b(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\b",
        "BITCOIN": r"(?:[13][A-KM-Za-km-z1-9]{25,34}|bc1[0-9A-Za-z]{25,90})",
        "COMMENT": r"//.*?$|/\*.*?\*/",
    }

    def __init__(self, logger: ILogger, settings: ISettingsManager) -> None:
        self._logger = logger
        self._settings = settings
        self._patterns: Dict[MatchType, re.Pattern] = {}
        self._load_patterns()
        settings.register_handler("patterns", self._on_patterns_changed)

    def analyze(self, text: str) -> AnalysisResult:
        """Perform text analysis with current patterns."""
        start_time = time.perf_counter()
        matches: List[PatternMatch] = []

        for kind, pattern in self._patterns.items():
            try:
                for match in pattern.finditer(text):
                    start = self._calculate_position(text, match.start())
                    end = self._calculate_position(text, match.end())
                    matches.append(PatternMatch(
                        text=match.group(),
                        kind=kind,
                        start=start,
                        end=end))
            except Exception as e:
                self._logger.log(
                    LogLevel.ERROR,
                    f"Pattern matching failed for {kind.name}",
                    error=str(e))

        elapsed = time.perf_counter() - start_time
        return AnalysisResult(
            matches=sorted(matches, key=lambda m: m.start.absolute),
            metrics=AnalysisMetrics(
                total_matches=len(matches),
                processing_time=elapsed))

    def _load_patterns(self) -> None:
        """Load patterns from configuration."""
        patterns = self._settings.get(
            "patterns", "default", self.DEFAULT_PATTERNS)
        self._patterns = {}
        for name, pattern_str in patterns.items():
            try:
                self._patterns[MatchType[name]] = re.compile(
                    pattern_str, re.MULTILINE | re.DOTALL)
            except (KeyError, re.error) as e:
                self._logger.log(
                    LogLevel.ERROR,
                    f"Invalid pattern '{name}': {pattern_str}",
                    error=str(e))
                if name in self.DEFAULT_PATTERNS:
                    self._logger.log(
                        LogLevel.WARNING,
                        f"Using default pattern for {name}")
                    self._patterns[MatchType[name]] = re.compile(
                        self.DEFAULT_PATTERNS[name],
                        re.MULTILINE | re.DOTALL)

    def _on_patterns_changed(self, key: str, value: Any) -> None:
        """Handle pattern configuration updates."""
        self._load_patterns()

    @staticmethod
    def _calculate_position(text: str, index: int) -> TextPosition:
        """Calculate line/column position from absolute index."""
        preceding = text[:index]
        line = preceding.count('\n') + 1
        last_newline = preceding.rfind('\n')
        column = index - last_newline if last_newline != -1 else index + 1
        return TextPosition(line, column, index)

# endregion

# region Presentation Layer


class SyntaxHighlighter:
    """Advanced syntax highlighting engine."""

    COLOR_SCHEME = {
        MatchType.TIME: QColor("#f9e2af"),
        MatchType.BITCOIN: QColor("#a6e3a1"),
        MatchType.COMMENT: QColor("#74c7ec"),
    }

    def __init__(self, editor: QPlainTextEdit) -> None:
        self._editor = editor
        self._selections: List[QTextEdit.ExtraSelection] = []

    def apply_highlights(self, matches: List[PatternMatch]) -> None:
        """Apply highlights to matched patterns."""
        self._selections.clear()
        for match in matches:
            cursor = QTextCursor(self._editor.document())
            cursor.setPosition(match.start.absolute)
            cursor.setPosition(
                match.end.absolute,
                QTextCursor.MoveMode.KeepAnchor)

            fmt = QTextCharFormat()
            fmt.setForeground(self.COLOR_SCHEME[match.kind])

            selection = QTextEdit.ExtraSelection()
            selection.format = fmt  # type:ignore
            selection.cursor = cursor  # type:ignore
            self._selections.append(selection)

        self._editor.setExtraSelections(self._selections)


class AnalysisResultsView(QTabWidget):
    """Interactive results visualization component."""

    def __init__(self) -> None:
        super().__init__()
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI components."""
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Тип", "Текст", "Строка", "Столбец", "Диапазон"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.addTab(self._table, "Совпадения")

    def update_results(self, result: AnalysisResult) -> None:
        """Update display with new analysis results."""
        self._table.setRowCount(len(result.matches))
        for row, match in enumerate(result.matches):
            self._table.setItem(row, 0, QTableWidgetItem(match.kind.name))
            self._table.setItem(row, 1, QTableWidgetItem(match.text))
            self._table.setItem(row, 2, QTableWidgetItem(str(match.start.line)))
            self._table.setItem(row, 3, QTableWidgetItem(str(match.start.column)))
            self._table.setItem(row, 4, QTableWidgetItem(f"{match.start.absolute}-{match.end.absolute}"))

# endregion

# region Configuration


class SettingsManager(QSettings):
    """Advanced configuration management system."""

    setting_changed = Signal(str, str, object)

    def __init__(self) -> None:
        super().__init__("TextAnalyzerPro", "Settings")
        self._handlers: DefaultDict[str, List[Callable[[str, Any], None]]] = defaultdict(list)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.value(f"{section}/{key}", default)

    def set(self, section: str, key: str, value: Any) -> None:
        self.setValue(f"{section}/{key}", value)
        self.setting_changed.emit(section, key, value)
        self._notify_handlers(section, key, value)

    def register_handler(self, section: str,
                         handler: Callable[[str, Any], None]) -> None:
        self._handlers[section].append(handler)

    def _notify_handlers(self, section: str, key: str, value: Any) -> None:
        for handler in self._handlers.get(section, []):
            handler(key, value)


class SettingsDialog(QDialog):
    """Configuration dialog for application settings."""

    def __init__(self, settings: ISettingsManager):
        super().__init__()
        self._settings = settings
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize dialog components."""
        self.setWindowTitle("Настройки")
        layout = QFormLayout(self)

        # Pattern settings
        self.pattern_edits = {}
        default_patterns = RegexAnalyzer.DEFAULT_PATTERNS
        patterns = self._settings.get("patterns", "default", default_patterns)
        for name in MatchType.__members__:
            edit = QLineEdit(patterns.get(name, ""))
            self.pattern_edits[name] = edit
            layout.addRow(f"{name} Pattern:", edit)

        # UI settings
        self.font_spin = QSpinBox()
        self.font_spin.setValue(int(self._settings.get("ui", "font_size", 12)))
        layout.addRow("Font Size:", self.font_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save_settings)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _save_settings(self) -> None:
        """Save updated settings to configuration."""
        new_patterns = {name: edit.text() for name, edit in self.pattern_edits.items()}
        self._settings.set("patterns", "default", new_patterns)
        self._settings.set("ui", "font_size", self.font_spin.value())
        self.accept()

# endregion

# region Application Core


class ApplicationModel(QObject):
    """Central data model for application state."""

    content_changed = Signal(str)
    analysis_completed = Signal(AnalysisResult)

    def __init__(self) -> None:
        super().__init__()
        self._content = ""
        self._last_result: Optional[AnalysisResult] = None

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        if value != self._content:
            self._content = value
            self.content_changed.emit(value)

    def update_result(self, result: AnalysisResult) -> None:
        self._last_result = result
        self.analysis_completed.emit(result)


class ApplicationController(QObject):
    """Orchestrates application workflow and business logic."""

    def __init__(
        self,
        model: ApplicationModel,
        view: IView,
        file_handler: IFileHandler,
        analyzer: ITextAnalyzer,
        logger: ILogger,
        settings: ISettingsManager
    ) -> None:
        super().__init__()
        self._model = model
        self._view = view
        self._file_handler = file_handler
        self._analyzer = analyzer
        self._logger = logger
        self._settings = settings

        self._connect_signals()
        self._register_settings_handlers()

    def _connect_signals(self) -> None:
        """Connect component signals to controller slots."""
        self._view.content_changed.connect(self._on_content_changed)
        self._view.file_open_requested.connect(self._handle_file_open)
        self._model.analysis_completed.connect(self._on_analysis_completed)

    def _register_settings_handlers(self) -> None:
        """Register configuration change handlers."""
        self._settings.register_handler("ui", self._handle_ui_changes)

    @Slot(str)
    def _on_content_changed(self, content: str) -> None:
        """Handle text content changes from editor."""
        try:
            result = self._analyzer.analyze(content)
            self._model.update_result(result)
        except Exception as e:
            self._logger.log(
                LogLevel.ERROR,
                "Analysis failed",
                error=str(e),
                traceback=traceback.format_exc())

    @Slot(str)
    def _handle_file_open(self, path: str) -> None:
        """Handle file open requests."""
        try:
            content = self._file_handler.read(Path(path))
            self._model.content = content
        except Exception as e:
            self._logger.log(
                LogLevel.ERROR,
                "File open failed",
                path=path,
                error=str(e))
            QMessageBox.critical(
                self._view,
                "File Error",
                f"Failed to open file: {e}")

    @Slot(AnalysisResult)
    def _on_analysis_completed(self, result: AnalysisResult) -> None:
        """Update view with analysis results."""
        self._view.display_results(result)

    def _handle_ui_changes(self, key: str, value: Any) -> None:
        """Update UI components based on settings changes."""
        if key == "theme":
            self._apply_theme(value)
        elif key == "font_size":
            self._adjust_font_size(value)

    def _apply_theme(self, theme: str) -> None:
        """Apply visual theme to UI components."""
        self._logger.log(LogLevel.INFO, f"Theme changed to {theme}")

    def _adjust_font_size(self, size: int) -> None:
        """Adjust UI font sizes."""
        self._logger.log(LogLevel.INFO, f"Font size changed to {size}")

# endregion

# region Main Window


class MainWindow(QMainWindow):
    """Primary application window implementing IView."""

    content_changed = Signal(str)
    file_open_requested = Signal(str)

    def __init__(self, theme_manager: ThemeManager, settings: ISettingsManager, logger: ILogger):
        super().__init__()
        self._theme_manager = theme_manager
        self._theme_manager.theme_changed.connect(self._on_theme_changed)
        self._settings = settings
        self._logger = logger
        self._editor = QPlainTextEdit()
        self._results_view = AnalysisResultsView()
        self._highlighter = SyntaxHighlighter(self._editor)
        self._init_ui()
        self._create_menu()

    def _init_ui(self) -> None:
        """Initialize window layout and components."""
        self.setWindowTitle("Анализатор")
        self.resize(1280, 720)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._editor, 70)
        layout.addWidget(self._results_view, 30)
        self.setCentralWidget(container)

        self._editor.textChanged.connect(
            lambda: self.content_changed.emit(self._editor.toPlainText()))

    def _create_menu(self) -> None:
        """Create application menu system."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&Файл")
        open_action = QAction("&Открыть", self)
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        # Settings menu
        settings_menu = menubar.addMenu("&Настройка")
        config_action = QAction("&Конфигурация", self)
        config_action.triggered.connect(self._show_settings)
        settings_menu.addAction(config_action)

    def _open_file(self) -> None:
        """Handle file open action."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Text File",
            "",
            "Text Files (*.txt *.log *.csv);;All Files (*)")
        if path:
            self.file_open_requested.emit(path)

    def _show_settings(self) -> None:
        """Display settings dialog."""
        dialog = SettingsDialog(self._settings)
        if dialog.exec():
            self._logger.log(LogLevel.INFO, "Settings updated")

    def display_results(self, result: AnalysisResult) -> None:
        """Update UI with analysis results."""
        self._results_view.update_results(result)
        self._highlighter.apply_highlights(result.matches)

    @Slot(ITheme)
    def _on_theme_changed(self, theme: ITheme) -> None:
        """Update theme-dependent components"""
        self._update_syntax_highlighting(theme.colors)

    def _update_syntax_highlighting(self, colors: Dict[str, QColor]) -> None:
        self._highlighter.COLOR_SCHEME = {
            MatchType.TIME: colors["yellow"],
            MatchType.BITCOIN: colors["green"],
            MatchType.COMMENT: colors["blue"],
        }

    def set_content(self, text: str) -> None:
        """Set editor content."""
        self._editor.setPlainText(text)

    def show(self) -> None:
        """Show main window."""
        super().show()

# endregion

# region Application Bootstrap


class TextAnalyzerApp(QApplication):
    """Custom application container with lifecycle management."""

    def __init__(self, args: List[str]) -> None:
        super().__init__(args)

        # Initialize core services first
        self.settings = SettingsManager()
        self.logger = StructuredLogger()

        # Setup theme management
        self._theme_manager = ThemeManager(self.settings)
        self._theme_manager.register_theme(CatppuccinDarkTheme())

        # Apply stored theme
        theme_name = self.settings.get("ui", "theme", "catppuccin-dark")
        self._theme_manager.set_theme(theme_name)

        # Initialize remaining components
        self.file_handler = SecureFileHandler(self.logger)
        self.analyzer = RegexAnalyzer(self.logger, self.settings)
        self.model = ApplicationModel()
        self.view = MainWindow(self._theme_manager, self.settings, self.logger)
        self.controller = ApplicationController(
            self.model,
            self.view,
            self.file_handler,
            self.analyzer,
            self.logger,
            self.settings
        )

        # Apply theme to application instance
        self._theme_manager.current_theme().apply(self)


def main() -> None:
    """Application entry point with error handling."""
    try:
        app = TextAnalyzerApp(sys.argv)
        app.view.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.critical(f"Fatal initialization error: {e}", exc_info=True)
        parent = QApplication.activeWindow()
        QMessageBox.critical(
            parent,
            "Critical Error",
            f"Application failed to start: {str(e)}",
            QMessageBox.StandardButton.Ok)
        sys.exit(1)


if __name__ == "__main__":
    main()

# endregion
