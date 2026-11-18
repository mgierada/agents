from http.server import BaseHTTPRequestHandler
from typing import Any, cast

import pytest

from cleanup.gallery.server import RunRegistry, make_handler


class HungUpSocket:
    """A wfile whose client already closed the connection."""

    def __init__(self, error: type[OSError]) -> None:
        self.error: type[OSError] = error

    def write(self, data: bytes) -> int:
        """Fail like a socket write to a closed peer."""
        raise self.error()


def handler_writing_to(wfile: Any) -> Any:
    """A Handler instance wired to `wfile`, without opening a real connection."""
    cls: type[BaseHTTPRequestHandler] = make_handler(cast(RunRegistry, None))
    handler: Any = cls.__new__(cls)
    handler.wfile = wfile
    handler.path = "/api/items"
    handler.requestline = "GET /api/items HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = "GET"
    handler.client_address = ("127.0.0.1", 0)
    return handler


@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError])
def test_send_json_ignores_client_that_hung_up(error: type[OSError]) -> None:
    handler: Any = handler_writing_to(HungUpSocket(error))

    handler.send_json({"ok": True})
