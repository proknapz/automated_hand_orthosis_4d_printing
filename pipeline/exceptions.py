"""Pipeline-specific errors."""


class HandNotDetectedError(Exception):
    """Raised when MediaPipe does not find a hand in the image."""


class PipelineError(Exception):
    """Raised for recoverable processing failures."""
