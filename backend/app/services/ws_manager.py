"""StoneSense-AI WebSocket Connection Manager.

Provides real-time broadcasting of federated learning events and status updates
to connected Developer and Hospital dashboards.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("StoneSenseWSManager")


class ConnectionManager:
    """Manages active WebSocket connections for real-time telemetry streaming."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total active connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcasts a JSON-serializable message dictionary to all active clients."""
        if not self.active_connections:
            return

        payload = json.dumps(message)
        dead_connections = set()

        async with self._lock:
            connections = list(self.active_connections)

        for connection in connections:
            try:
                await connection.send_text(payload)
            except Exception as e:
                logger.warning(f"Error sending WebSocket message: {e}")
                dead_connections.add(connection)

        if dead_connections:
            async with self._lock:
                for dead in dead_connections:
                    self.active_connections.discard(dead)

    def broadcast_sync(self, message: Dict[str, Any]):
        """Synchronous wrapper to broadcast messages from background threads."""
        if not self.active_connections:
            return

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
        except Exception:
            pass


ws_manager = ConnectionManager()

