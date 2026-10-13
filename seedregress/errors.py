"""Errors the CLI and web UI can show without a traceback."""


class SeedRegressError(Exception):
    """Base error for expected failures."""


class SuiteError(SeedRegressError):
    """The suite file or a case is not usable."""


class GpuConfirmationRequired(SeedRegressError):
    """A ComfyUI network call was attempted without an explicit GPU confirmation."""


class QueueBusy(SeedRegressError):
    """ComfyUI already has running or pending work, so nothing was submitted."""


class ComfyError(SeedRegressError):
    """ComfyUI returned an error or could not be reached."""


class AmbiguousSubmission(ComfyError):
    """A prompt request may have reached ComfyUI, but no prompt id came back."""

    def __init__(self, message: str, *, prompt_id: str | None = None):
        super().__init__(message)
        # A malformed response can still contain a value which looks like a
        # prompt id.  Keep it in the durable run outcome for manual
        # reconciliation, but never treat it as safe to cancel automatically.
        self.prompt_id = prompt_id


class TerminalPromptFailure(ComfyError):
    """ComfyUI recorded this prompt as terminally failed."""
