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
