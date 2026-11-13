class UnknownProviderError(ValueError):
    pass


class InvalidOutputError(ValueError):
    """The model answered, but not with output matching the requested schema. Worth a retry."""
