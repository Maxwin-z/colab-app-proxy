import logging
import json
import os
import signal
import shutil
import socket
from collections import deque
from datetime import datetime
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from starlette.background import BackgroundTask
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proxy")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI()

# ANSI color codes
class Colors:
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

COLOR_CYCLE = [Colors.CYAN, Colors.MAGENTA, Colors.GREEN, Colors.YELLOW]

# --- Dynamic Mappings Store ---
MAPPINGS_FILE = Path(__file__).parent / "mappings.json"

# Each mapping: {"path": "/xxx", "port": 8001, "name": "My Service"}
_mappings: list[dict] = []


def _load_mappings():
    global _mappings
    if MAPPINGS_FILE.exists():
        try:
            _mappings = json.loads(MAPPINGS_FILE.read_text())
        except Exception:
            _mappings = []
    else:
        _mappings = []


def _save_mappings():
    MAPPINGS_FILE.write_text(json.dumps(_mappings, indent=2))


def _get_http_target(path_prefix: str, port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _get_ws_target(path_prefix: str, port: int) -> str:
    return f"ws://127.0.0.1:{port}"


def _find_mapping(full_path: str):
    """Find the matching mapping for a request path. Returns (mapping, suffix) or (None, None)."""
    for m in _mappings:
        prefix = m["path"]
        if full_path == prefix or full_path.startswith(prefix + "/"):
            suffix = full_path[len(prefix):]
            if suffix.startswith("/"):
                suffix = suffix[1:]
            return m, suffix
    return None, None


def _get_color(index: int) -> str:
    return COLOR_CYCLE[index % len(COLOR_CYCLE)]


# Load mappings on startup
_load_mappings()

# --- Quick Deploy App Manager ---
APPS_DIR = Path(__file__).parent / "apps"
APPS_FILE = Path(__file__).parent / "apps_state.json"
PRESET_REPOS = [
    {"name": "VibeVoice-0.5B", "git_url": "https://github.com/Maxwin-z/VibeVoice-0.5B.git"},
    {"name": "Colab Ollama App", "git_url": "https://github.com/Maxwin-z/colab-ollama-app.git"},
    {"name": "Z-Image Turbo App", "git_url": "https://github.com/Maxwin-z/z-image-turbo-app.git"},
]

_apps: list[dict] = []
_app_logs: dict[str, deque] = {}
_app_tasks: dict[str, asyncio.Task] = {}  # log reader tasks
_app_monitor_tasks: dict[str, asyncio.Task] = {}  # process monitor tasks


def _load_apps():
    global _apps
    if APPS_FILE.exists():
        try:
            saved = json.loads(APPS_FILE.read_text())
            _apps = []
            for a in saved:
                a["status"] = "stopped"
                a["pid"] = None
                a["pgid"] = None
                a["ports"] = []
                a["error"] = None
                a["git_hash"] = _get_git_hash(APPS_DIR / a["id"])
                _apps.append(a)
        except Exception:
            _apps = []
    else:
        _apps = []


def _save_apps():
    persistent = []
    for a in _apps:
        persistent.append({
            "id": a["id"],
            "name": a["name"],
            "git_url": a["git_url"],
            "deployed_at": a.get("deployed_at"),
        })
    APPS_FILE.write_text(json.dumps(persistent, indent=2))


def _find_app(app_id: str) -> dict | None:
    return next((a for a in _apps if a["id"] == app_id), None)


def _app_dir(app: dict) -> Path:
    return APPS_DIR / app["id"]


def _get_app_log(app_id: str) -> deque:
    if app_id not in _app_logs:
        _app_logs[app_id] = deque(maxlen=2000)
    return _app_logs[app_id]


def _app_log_line(app_id: str, line: str):
    log = _get_app_log(app_id)
    ts = datetime.now().strftime("%H:%M:%S")
    log.append(f"[{ts}] {line}")


def _get_git_hash(app_dir: Path) -> str | None:
    """Read the short git commit hash from an app directory (sync, no subprocess)."""
    try:
        head = (app_dir / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref_path = app_dir / ".git" / head[5:]
            commit = ref_path.read_text().strip()
        else:
            commit = head
        return commit[:7]
    except Exception:
        return None


def _app_path(app: dict) -> str:
    """Return the proxy path for an app, e.g. /colab-ollama-app."""
    return "/" + app["id"]


def _sync_app_mapping(app: dict):
    """Create or update a proxy mapping for a running app with detected ports."""
    path = _app_path(app)
    if app["status"] == "running" and app.get("ports"):
        port = app["ports"][0]
        existing = next((m for m in _mappings if m["path"] == path), None)
        if existing:
            if existing["port"] != port:
                existing["port"] = port
                _save_mappings()
                logger.info(f"Updated app mapping: {path} -> 127.0.0.1:{port}")
        else:
            mapping = {"path": path, "port": port, "name": app["name"]}
            _mappings.append(mapping)
            _save_mappings()
            logger.info(f"Auto-mapped app: {path} -> 127.0.0.1:{port}")
    else:
        _remove_app_mapping(app)


def _remove_app_mapping(app: dict):
    """Remove the proxy mapping for an app."""
    global _mappings
    path = _app_path(app)
    before = len(_mappings)
    _mappings = [m for m in _mappings if m["path"] != path]
    if len(_mappings) < before:
        _save_mappings()
        logger.info(f"Removed app mapping: {path}")


def _merge_apps_with_presets() -> list[dict]:
    """Merge preset repos with deployed apps. Presets not yet deployed appear as stubs."""
    result = list(_apps)
    deployed_urls = {a["git_url"] for a in _apps}
    for preset in PRESET_REPOS:
        if preset["git_url"] not in deployed_urls:
            result.append({
                "id": _url_to_id(preset["git_url"]),
                "name": preset["name"],
                "git_url": preset["git_url"],
                "status": "stopped",
                "pid": None,
                "pgid": None,
                "ports": [],
                "error": None,
                "deployed_at": None,
                "preset": True,
            })
    return result


def _url_to_id(git_url: str) -> str:
    """Derive a stable ID from a git URL."""
    name = git_url.rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


async def _start_app(app: dict):
    """Start an app by running bash start.sh in its directory."""
    app_dir = _app_dir(app)
    start_script = app_dir / "start.sh"
    if not start_script.exists():
        app["status"] = "error"
        app["error"] = "start.sh not found"
        _app_log_line(app["id"], "ERROR: start.sh not found in repo")
        return

    app["status"] = "running"
    app["error"] = None
    _app_log_line(app["id"], "Starting app...")

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", "start.sh",
            cwd=str(app_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        app["pid"] = proc.pid
        try:
            app["pgid"] = os.getpgid(proc.pid)
        except Exception:
            app["pgid"] = proc.pid

        _app_log_line(app["id"], f"Process started (PID={proc.pid}, PGID={app['pgid']})")

        # Start log reader task
        task = asyncio.create_task(_read_app_logs(app, proc))
        _app_tasks[app["id"]] = task

        # Start monitor task
        monitor = asyncio.create_task(_monitor_app(app, proc))
        _app_monitor_tasks[app["id"]] = monitor

    except Exception as e:
        app["status"] = "error"
        app["error"] = str(e)
        _app_log_line(app["id"], f"ERROR starting: {e}")


async def _read_app_logs(app: dict, proc: asyncio.subprocess.Process):
    """Read stdout/stderr lines from process into log deque."""
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            _app_log_line(app["id"], text)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        _app_log_line(app["id"], f"Log reader error: {e}")


async def _monitor_app(app: dict, proc: asyncio.subprocess.Process):
    """Watch process exit and update status."""
    try:
        await proc.wait()
        exit_code = proc.returncode
        _app_log_line(app["id"], f"Process exited with code {exit_code}")
        if app["status"] == "running":
            app["status"] = "stopped"
        app["pid"] = None
        app["pgid"] = None
        app["ports"] = []
        _remove_app_mapping(app)
    except asyncio.CancelledError:
        pass


async def _stop_app(app: dict):
    """Stop a running app. SIGTERM → wait → SIGKILL."""
    pgid = app.get("pgid")
    pid = app.get("pid")
    if not pgid and not pid:
        app["status"] = "stopped"
        return

    _app_log_line(app["id"], "Stopping app...")

    # Cancel log reader and monitor tasks
    for tasks_dict in (_app_tasks, _app_monitor_tasks):
        task = tasks_dict.pop(app["id"], None)
        if task:
            task.cancel()

    try:
        os.killpg(pgid, signal.SIGTERM)
        _app_log_line(app["id"], f"Sent SIGTERM to process group {pgid}")
    except ProcessLookupError:
        pass
    except Exception as e:
        _app_log_line(app["id"], f"SIGTERM error: {e}")

    # Wait up to 5 seconds for process to die
    await asyncio.sleep(5)

    try:
        os.killpg(pgid, signal.SIGKILL)
        _app_log_line(app["id"], f"Sent SIGKILL to process group {pgid}")
    except ProcessLookupError:
        pass
    except Exception:
        pass

    app["status"] = "stopped"
    app["pid"] = None
    app["pgid"] = None
    app["ports"] = []
    _remove_app_mapping(app)
    _app_log_line(app["id"], "App stopped")


async def _detect_ports_loop():
    """Background loop: detect listening ports for running apps."""
    while True:
        await asyncio.sleep(5)
        for app in _apps:
            if app["status"] != "running" or not app.get("pgid"):
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    "lsof", "-i", "-P", "-n", "-sTCP:LISTEN",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await proc.communicate()
                ports = set()
                pgid = app["pgid"]
                for line in stdout.decode(errors="replace").splitlines()[1:]:
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    try:
                        line_pid = int(parts[1])
                        if os.getpgid(line_pid) == pgid:
                            addr = parts[8]
                            if ":" in addr:
                                port = int(addr.rsplit(":", 1)[1])
                                ports.add(port)
                    except (ProcessLookupError, ValueError):
                        continue
                new_ports = sorted(ports)
                if new_ports != app["ports"]:
                    app["ports"] = new_ports
                    _sync_app_mapping(app)
            except Exception:
                pass


_load_apps()
APPS_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_detect_ports_loop())


# --- Management API (under /_api/) ---

@app.get("/_api/mappings")
async def api_list_mappings():
    return _mappings


@app.post("/_api/mappings")
async def api_add_mapping(body: dict):
    path = body.get("path", "").strip()
    port = body.get("port")
    name = body.get("name", "").strip()

    if not path or not path.startswith("/"):
        raise HTTPException(400, "Path must start with /")
    if not port or not isinstance(port, int) or port < 1 or port > 65535:
        raise HTTPException(400, "Port must be 1-65535")

    # Reject reserved prefixes
    if path.startswith("/_api"):
        raise HTTPException(400, "Path /_api is reserved")

    # Check duplicate
    for m in _mappings:
        if m["path"] == path:
            raise HTTPException(409, f"Path {path} already exists")

    mapping = {"path": path, "port": port, "name": name or path.lstrip("/")}
    _mappings.append(mapping)
    _save_mappings()
    logger.info(f"Added mapping: {path} -> 127.0.0.1:{port}")
    return mapping


@app.delete("/_api/mappings/{path:path}")
async def api_delete_mapping(path: str):
    global _mappings
    if not path.startswith("/"):
        path = "/" + path
    before = len(_mappings)
    _mappings = [m for m in _mappings if m["path"] != path]
    if len(_mappings) == before:
        raise HTTPException(404, "Mapping not found")
    _save_mappings()
    logger.info(f"Deleted mapping: {path}")
    return {"ok": True}


@app.get("/_api/port-status/{path:path}")
async def api_port_status(path: str):
    if not path.startswith("/"):
        path = "/" + path
    mapping = next((m for m in _mappings if m["path"] == path), None)
    if not mapping:
        raise HTTPException(404, "Mapping not found")

    port = mapping["port"]
    online = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", port))
        online = result == 0
        sock.close()
    except Exception:
        pass

    return {"path": path, "port": port, "online": online}


# --- Dashboard ---

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text())


# --- Quick Deploy API ---

@app.get("/_api/apps")
async def api_list_apps():
    merged = _merge_apps_with_presets()
    return [{
        "id": a["id"],
        "name": a["name"],
        "git_url": a["git_url"],
        "status": a["status"],
        "path": _app_path(a),
        "ports": a.get("ports", []),
        "error": a.get("error"),
        "deployed_at": a.get("deployed_at"),
        "git_hash": a.get("git_hash"),
        "preset": a.get("preset", False),
    } for a in merged]


@app.post("/_api/apps/deploy")
async def api_deploy_app(body: dict):
    git_url = body.get("git_url", "").strip()
    name = body.get("name", "").strip()

    if not git_url:
        raise HTTPException(400, "git_url is required")

    app_id = _url_to_id(git_url)
    if not name:
        name = app_id

    # Check if already exists
    app = _find_app(app_id)
    if app and app["status"] in ("running", "deploying"):
        raise HTTPException(409, "App is already running or deploying")

    if not app:
        app = {
            "id": app_id,
            "name": name,
            "git_url": git_url,
            "status": "deploying",
            "pid": None,
            "pgid": None,
            "ports": [],
            "error": None,
            "deployed_at": None,
        }
        _apps.append(app)
    else:
        app["status"] = "deploying"
        app["error"] = None

    _save_apps()

    # Run clone/pull + start in background
    asyncio.create_task(_deploy_and_start(app))

    return {"id": app_id, "status": "deploying"}


async def _deploy_and_start(app: dict):
    """Clone or pull the repo, then start the app."""
    app_dir = _app_dir(app)
    git_url = app["git_url"]
    app_id = app["id"]

    try:
        if app_dir.exists() and (app_dir / ".git").exists():
            _app_log_line(app_id, f"Pulling latest from {git_url}...")
            proc = await asyncio.create_subprocess_exec(
                "git", "pull",
                cwd=str(app_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            for line in stdout.decode(errors="replace").splitlines():
                _app_log_line(app_id, line)
            if proc.returncode != 0:
                app["status"] = "error"
                app["error"] = "git pull failed"
                _app_log_line(app_id, "ERROR: git pull failed")
                return
        else:
            _app_log_line(app_id, f"Cloning {git_url}...")
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", git_url, str(app_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            for line in stdout.decode(errors="replace").splitlines():
                _app_log_line(app_id, line)
            if proc.returncode != 0:
                app["status"] = "error"
                app["error"] = "git clone failed"
                _app_log_line(app_id, "ERROR: git clone failed")
                return

        app["deployed_at"] = datetime.now().isoformat()
        app["git_hash"] = _get_git_hash(app_dir)
        _save_apps()
        _app_log_line(app_id, f"Deploy complete (commit: {app['git_hash']}), starting app...")

        await _start_app(app)

    except Exception as e:
        app["status"] = "error"
        app["error"] = str(e)
        _app_log_line(app_id, f"Deploy error: {e}")


@app.post("/_api/apps/{app_id}/start")
async def api_start_app(app_id: str):
    app = _find_app(app_id)
    if not app:
        raise HTTPException(404, "App not found")
    if app["status"] in ("running", "deploying"):
        raise HTTPException(409, f"App is already {app['status']}")
    if not _app_dir(app).exists():
        raise HTTPException(400, "App not deployed yet. Use deploy first.")

    asyncio.create_task(_start_app(app))
    return {"id": app_id, "status": "starting"}


@app.post("/_api/apps/{app_id}/update")
async def api_update_app(app_id: str):
    app = _find_app(app_id)
    if not app:
        raise HTTPException(404, "App not found")
    if app["status"] in ("deploying", "updating"):
        raise HTTPException(409, f"App is already {app['status']}")
    app_dir = _app_dir(app)
    if not app_dir.exists():
        raise HTTPException(400, "App not deployed yet. Use deploy first.")

    # Stop first if running
    was_running = app["status"] == "running"
    if was_running:
        await _stop_app(app)

    app["status"] = "updating"
    app["error"] = None
    asyncio.create_task(_update_app(app, restart=was_running))
    return {"id": app_id, "status": "updating"}


async def _update_app(app: dict, restart: bool = False):
    app_dir = _app_dir(app)
    app_id = app["id"]
    try:
        _app_log_line(app_id, "Pulling latest changes...")
        proc = await asyncio.create_subprocess_exec(
            "git", "pull",
            cwd=str(app_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        for line in stdout.decode(errors="replace").splitlines():
            _app_log_line(app_id, line)
        if proc.returncode != 0:
            app["status"] = "error"
            app["error"] = "git pull failed"
            _app_log_line(app_id, "ERROR: git pull failed")
            return

        app["git_hash"] = _get_git_hash(app_dir)
        _save_apps()
        _app_log_line(app_id, f"Update complete (commit: {app['git_hash']})")
        if restart:
            _app_log_line(app_id, "Restarting app...")
            await _start_app(app)
        else:
            app["status"] = "stopped"
    except Exception as e:
        app["status"] = "error"
        app["error"] = str(e)
        _app_log_line(app_id, f"Update error: {e}")


@app.post("/_api/apps/{app_id}/stop")
async def api_stop_app(app_id: str):
    app = _find_app(app_id)
    if not app:
        raise HTTPException(404, "App not found")
    if app["status"] != "running":
        raise HTTPException(400, "App is not running")

    await _stop_app(app)
    return {"id": app_id, "status": "stopped"}


@app.delete("/_api/apps/{app_id}")
async def api_delete_app(app_id: str):
    global _apps
    app = _find_app(app_id)
    if not app:
        raise HTTPException(404, "App not found")

    # Stop if running
    if app["status"] == "running":
        await _stop_app(app)

    # Remove directory
    app_dir = _app_dir(app)
    if app_dir.exists():
        shutil.rmtree(app_dir, ignore_errors=True)
        _app_log_line(app_id, "App directory deleted")

    # Remove from list
    _apps = [a for a in _apps if a["id"] != app_id]
    _save_apps()

    # Clean up logs
    _app_logs.pop(app_id, None)

    return {"ok": True}


@app.get("/_api/apps/{app_id}/logs")
async def api_get_app_logs(app_id: str, offset: int = 0, limit: int = 200):
    log = _get_app_log(app_id)
    lines = list(log)
    total = len(lines)
    sliced = lines[offset:offset + limit]
    return {"lines": sliced, "total": total, "offset": offset}


@app.get("/_api/apps/{app_id}/readme")
async def api_get_app_readme(app_id: str):
    app = _find_app(app_id)
    app_dir = APPS_DIR / app_id
    if not app_dir.exists():
        raise HTTPException(404, "App not deployed")
    # Try common README filenames
    for name in ("README.md", "readme.md", "Readme.md", "README.MD"):
        readme = app_dir / name
        if readme.exists():
            return {"content": readme.read_text(encoding="utf-8", errors="replace")}
    raise HTTPException(404, "README.md not found")


# --- HTTP Proxy ---

async def proxy_http_request(client: httpx.AsyncClient, request: Request, target_url: str, color: str):
    url = httpx.URL(target_url)
    url = url.copy_with(query=request.url.query.encode("utf-8"))

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    try:
        req = client.build_request(
            request.method,
            url,
            headers=headers,
            content=request.stream(),
            timeout=60.0
        )

        r = await client.send(req, stream=True)

        logger.info(f"{color}{request.method} {request.url.path} -> {target_url} [{r.status_code}]{Colors.RESET}")

        if r.status_code == 206 or "content-length" in r.headers:
            content = await r.aread()
            await r.aclose()

            response_headers = {}
            hop_by_hop = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade"}
            for key, value in r.headers.items():
                if key.lower() not in hop_by_hop:
                    response_headers[key] = value

            response_headers["content-length"] = str(len(content))

            return Response(
                content=content,
                status_code=r.status_code,
                headers=response_headers,
            )

        response_headers = {}
        hop_by_hop = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "content-length"}
        for key, value in r.headers.items():
            if key.lower() not in hop_by_hop:
                response_headers[key] = value

        return StreamingResponse(
            r.aiter_raw(),
            status_code=r.status_code,
            headers=response_headers,
            background=BackgroundTask(r.aclose)
        )
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return {"error": str(e), "status": 502}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def route_proxy(path: str, request: Request):
    full_path = request.url.path

    mapping, suffix = _find_mapping(full_path)
    if mapping:
        target_base = _get_http_target(mapping["path"], mapping["port"])
        target_dest = f"{target_base}/{suffix}" if suffix else target_base
        idx = _mappings.index(mapping)
        color = _get_color(idx)
        async with httpx.AsyncClient() as client:
            return await proxy_http_request(client, request, target_dest, color)

    return {"message": "Not found", "path": full_path}


# --- WebSocket Proxy ---

@app.websocket("/{path:path}")
async def websocket_proxy(websocket: WebSocket, path: str):
    full_path = websocket.url.path

    mapping, suffix = _find_mapping(full_path)

    if mapping:
        target_base = _get_ws_target(mapping["path"], mapping["port"])
        target_dest = f"{target_base}/{suffix}" if suffix else target_base

        query_string = websocket.url.query
        if query_string:
            target_dest = f"{target_dest}?{query_string}"

        await websocket.accept()

        try:
            async with websockets.connect(target_dest) as ws_target:

                async def forward_to_target():
                    try:
                        while True:
                            data = await websocket.receive_text()
                            await ws_target.send(data)
                    except Exception:
                        pass

                async def forward_to_client():
                    try:
                        while True:
                            data = await ws_target.recv()
                            await websocket.send_text(data)
                    except Exception:
                        pass

                await asyncio.gather(forward_to_target(), forward_to_client())

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            await websocket.close(code=1011)
    else:
        await websocket.close(code=4000)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
