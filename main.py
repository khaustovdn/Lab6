# region Imports
import re
import sys
import logging
import traceback
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QFont,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
# endregion

# region Core Types
T = TypeVar("T")
PathLike = TypeVar("PathLike", str, Path)


class MatchType(Enum):
    TIME = "time"
    BITCOIN = "bitcoin"
    COMMENT = "comment"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
# endregion

# region Exceptions


class ApplicationError(Exception):
    """Base application exception"""


class FileOperationError(ApplicationError):
    """File system operations failure"""


class AnalysisFailure(ApplicationError):
    """Text analysis processing error"""


# endregion

# region Data Models


@dataclass(frozen=True)
class TextPosition:
    line: int
    column: int
    absolute: int


@dataclass(frozen=True)
class PatternMatch:
    text: str
    type: MatchType
    start: TextPosition
    end: TextPosition


@dataclass(frozen=True)
class AnalysisResult:
    matches: List[PatternMatch]
    metrics: Dict[str, Any]


# endregion

# region Protocols


@runtime_checkable
class Logger(Protocol):
    def log(
        self,
        level: LogLevel,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...


@runtime_checkable
class FileHandler(Protocol):
    def read(self, path: Path) -> str:
        ...

    def write(self, path: Path, content: str) -> None:
        ...


@runtime_checkable
class TextAnalyzer(Protocol):
    def analyze(self, text: str) -> AnalysisResult:
        ...


# endregion

# region Service Layer


class AdvancedLogger(Logger):
    def __init__(self, handlers: Optional[List[logging.Handler]] = None):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

        if not handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            handlers = [handler]

        for h in handlers:
            self._logger.addHandler(h)

    def log(
        self,
        level: LogLevel,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        log_method = {
            LogLevel.DEBUG: self._logger.debug,
            LogLevel.INFO: self._logger.info,
            LogLevel.WARNING: self._logger.warning,
            LogLevel.ERROR: self._logger.error,
        }[level]

        extra = {"metadata": metadata} if metadata else {}
        log_method(message, extra=extra)


class SecureFileHandler(FileHandler):
    def __init__(self, logger: Logger):
        self.logger = logger

    def read(self, path: Path) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as f:
                content = f.read()
                self.logger.log(
                    LogLevel.INFO,
                    "File read successfully",
                    {"path": str(path), "size": len(content)},
                )
                return content
        except (IOError, UnicodeDecodeError) as e:
            self.logger.log(
                LogLevel.ERROR,
                "File read failure",
                {"path": str(path), "error": str(e)},
            )
            raise FileOperationError(f"Read error: {e}") from e

    def write(self, path: Path, content: str) -> None:
        try:
            with open(path, "w", encoding="utf-8", errors="strict") as f:
                f.write(content)
                self.logger.log(
                    LogLevel.INFO,
                    "File write successful",
                    {"path": str(path), "size": len(content)},
                )
        except (IOError, UnicodeEncodeError) as e:
            self.logger.log(
                LogLevel.ERROR,
                "File write failure",
                {"path": str(path), "error": str(e)},
            )
            raise FileOperationError(f"Write error: {e}") from e


class BaseTextAnalyzer(TextAnalyzer, ABC):
    @abstractmethod
    def analyze(self, text: str) -> AnalysisResult:
        ...


class RegexAnalyzer(BaseTextAnalyzer):
    PATTERNS = {
        MatchType.TIME: re.compile(
            r"\b(?:[01]\d|2[0-3]):(?:[0-5]\d):(?:[0-5]\d)\b"
        ),
        MatchType.BITCOIN: re.compile(
            r"(?:[13][a-km-zA-HJ-NP-Z1-9]{25,34}|"
            r"bc1[a-zA-HJ-NP-Z0-9]{25,90})"
        ),
        MatchType.COMMENT: re.compile(
            r"(//.*?$|/\*.*?\*/)", re.MULTILINE | re.DOTALL
        ),
    }

    def __init__(self, logger: Logger):
        self.logger = logger

    def analyze(self, text: str) -> AnalysisResult:
        try:
            matches = []

            for match_type, pattern in self.PATTERNS.items():
                for match in pattern.finditer(text):
                    start = self._get_text_position(text, match.start())
                    end = self._get_text_position(text, match.end())
                    matches.append(
                        PatternMatch(
                            text=match.group(),
                            type=match_type,
                            start=start,
                            end=end,
                        )
                    )

            return AnalysisResult(
                matches=sorted(matches, key=lambda m: m.start.absolute),
                metrics={"match_count": len(matches)},
            )
        except Exception as e:
            self.logger.log(
                LogLevel.ERROR,
                "Analysis failure",
                {"error": str(e)},
            )
            raise AnalysisFailure("Analysis failed") from e

    @staticmethod
    def _get_text_position(text: str, pos: int) -> TextPosition:
        if pos < 0 or pos > len(text):
            return TextPosition(1, 1, pos)

        lines = text[:pos].split("\n")
        return TextPosition(
            line=len(lines),
            column=len(lines[-1]) + 1,
            absolute=pos,
        )


# endregion

# region Presentation Layer


class MatchHighlighter:
    _COLOR_MAP = {
        MatchType.TIME: "#FFD700",
        MatchType.BITCOIN: "#90EE90",
        MatchType.COMMENT: "#ADD8E6",
    }

    def __init__(self, editor: QPlainTextEdit):
        self.editor = editor
        self._selections = []

    def apply_highlights(self, matches: List[PatternMatch]):
        self._selections.clear()
        for match in matches:
            selection = QTextEdit.ExtraSelection()
            selection.format = self._get_style(match.type)
            cursor = self.editor.textCursor()
            cursor.setPosition(match.start.absolute)
            cursor.setPosition(
                match.end.absolute, QTextCursor.MoveMode.KeepAnchor
            )
            selection.cursor = cursor
            self._selections.append(selection)
        self.editor.setExtraSelections(self._selections)

    def _get_style(self, match_type: MatchType) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(self._COLOR_MAP[match_type]))
        return fmt


class CodeEditor(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self._init_editor_settings()
        self.highlighter = MatchHighlighter(self)

    def _init_editor_settings(self):
        self.setFont(QFont("Fira Code", 12))
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

    def highlight_matches(self, matches: List[PatternMatch]):
        self.highlighter.apply_highlights(matches)


class ResultsTable(QTableWidget):
    def __init__(self, headers: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_table(headers)

    def _init_table(self, headers: List[str]):
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def update_data(self, data: List[List[Any]]):
        self.setRowCount(len(data))
        for row, items in enumerate(data):
            for col, item in enumerate(items):
                self.setItem(row, col, QTableWidgetItem(str(item)))


class ResultsPresenter(QTabWidget):
    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        self.match_table = ResultsTable([
            "Type", "Text", "Line", "Column", "Position"
        ])
        self.addTab(self.match_table, "Matches")

    def display_analysis(self, result: AnalysisResult):
        self.match_table.update_data([
            [
                match.type.value,
                match.text,
                match.start.line,
                match.start.column,
                f"{match.start.absolute}-{match.end.absolute}",
            ]
            for match in result.matches
        ])


# endregion

# region Application Core


class ApplicationModel(QObject):
    code_updated = Signal(str)
    analysis_completed = Signal(AnalysisResult)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._code = ""
        self._last_result: Optional[AnalysisResult] = None

    @property
    def code(self) -> str:
        return self._code

    @code.setter
    def code(self, value: str):
        if self._code != value:
            self._code = value
            self.code_updated.emit(value)

    def update_analysis(self, result: AnalysisResult):
        self._last_result = result
        self.analysis_completed.emit(result)


class ApplicationController(QObject):
    analyze_requested = Signal()
    file_open_requested = Signal(Path)
    file_save_requested = Signal(Path)
    application_exit = Signal()

    def __init__(
        self,
        model: ApplicationModel,
        file_handler: FileHandler,
        analyzer: TextAnalyzer,
        logger: Logger,
    ):
        super().__init__()
        self.model = model
        self.file_handler = file_handler
        self.analyzer = analyzer
        self.logger = logger
        self._connect_signals()

    def _connect_signals(self):
        self.model.analysis_completed.connect(self._handle_analysis_result)
        self.analyze_requested.connect(self._perform_analysis)
        self.file_open_requested.connect(self._handle_file_open)
        self.file_save_requested.connect(self._handle_file_save)

    @Slot(Path)
    def _handle_file_open(self, path: Path):
        try:
            self.model.code = self.file_handler.read(path)
        except FileOperationError as e:
            full_error = f"File operation failed: {str(e)}\nPath: {path}"
            self.logger.log(
                LogLevel.ERROR,
                full_error,
                {"path": str(path), "error": str(e)}
            )
            self._show_error(
                "File Error",
                f"Failed to process file:\n{path.name}\n\nDetails: {str(e)}"
            )

    @Slot(Path)
    def _handle_file_save(self, path: Path):
        try:
            self.file_handler.write(path, self.model.code)
        except FileOperationError as e:
            self.logger.log(LogLevel.ERROR, str(e))
            self._show_error("File Save Error", str(e))

    @Slot()
    def _perform_analysis(self):
        try:
            result = self.analyzer.analyze(self.model.code)
            self.model.update_analysis(result)
        except AnalysisFailure as e:
            error_msg = f"Analysis failure: {str(e)}"
            self.logger.log(
                LogLevel.ERROR,
                error_msg,
                {"error": str(e), "stack_trace": traceback.format_exc()}
            )
            self._show_error("Analysis Error", error_msg)

    @Slot(AnalysisResult)
    def _handle_analysis_result(self, result: AnalysisResult):
        self.logger.log(
            LogLevel.INFO,
            "Analysis completed",
            {"matches": len(result.matches)},
        )

    def _show_error(self, title: str, message: str):
        QMessageBox.critical(
            None,
            title,
            message,
            QMessageBox.StandardButton.Ok,
        )

    def _show_warning(self, title: str, message: str):
        QMessageBox.warning(
            None,
            title,
            message,
            QMessageBox.StandardButton.Ok,
        )


# endregion

# region UI Layer


class MainWindow(QMainWindow):
    def __init__(self, controller: ApplicationController):
        super().__init__()
        self.controller = controller
        self._init_ui()
        self._setup_menu()
        self._connect_signals()

    def _init_ui(self):
        self.setWindowTitle("Text Analyzer Pro")
        self.editor = CodeEditor()
        self.results = ResultsPresenter()

        layout = QVBoxLayout()
        layout.addWidget(self.editor, 70)
        layout.addWidget(self.results, 30)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(1280, 720)

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        open_action = QAction("&Open...", self)
        open_action.triggered.connect(self._handle_open)
        file_menu.addAction(open_action)

        save_action = QAction("&Save As...", self)
        save_action.triggered.connect(self._handle_save)
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        analyze_action = QAction("&Run Analysis", self)
        analyze_action.triggered.connect(self.controller.analyze_requested)
        analysis_menu.addAction(analyze_action)

    def _connect_signals(self):
        self.editor.textChanged.connect(self._update_model_code)
        self.controller.model.analysis_completed.connect(
            self._handle_analysis_result
        )

    def _update_model_code(self):
        self.controller.model.code = self.editor.toPlainText()

    def _handle_analysis_result(self, result: AnalysisResult):
        self.editor.highlight_matches(result.matches)
        self.results.display_analysis(result)

    def _handle_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Text File",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if path:
            self.controller.file_open_requested.emit(Path(path))

    def _handle_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Text File",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if path:
            self.controller.file_save_requested.emit(Path(path))

    def closeEvent(self, event: QCloseEvent):
        self.controller.application_exit.emit()
        super().closeEvent(event)


# endregion

# region Factory


class ApplicationFactory:
    @staticmethod
    def create() -> Tuple[MainWindow, ApplicationController]:
        logger = AdvancedLogger()
        file_handler = SecureFileHandler(logger)
        analyzer = RegexAnalyzer(logger)
        model = ApplicationModel()
        controller = ApplicationController(
            model=model,
            file_handler=file_handler,
            analyzer=analyzer,
            logger=logger,
        )
        window = MainWindow(controller)
        if app := QApplication.instance():
            controller.application_exit.connect(app.quit)
        return window, controller


# endregion

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window: Optional[QWidget] = None

    try:
        window, _ = ApplicationFactory.create()
        main_window = window
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(
            main_window or QWidget(),
            "Critical Error",
            f"Application failed to start:\n{str(e)}",
            QMessageBox.StandardButton.Ok,
        )
        sys.exit(1)
