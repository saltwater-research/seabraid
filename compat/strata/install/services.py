# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid installer, stage 5 (SERVICES): systemd --user units, opt-in install, health check.  UNTESTED-ON-HARDWARE.

OWNER     : this file owns the unit text (ordering, env, bind address), the install/enable steps and the health probe.
            It decides nothing about placement: roles, shm names and device ids come from the Plan; args and env
            come from config's output, taken DUCK-TYPED through adapt_config() (provisional contract below).
CONTRACT  : render_units(plan, cfg, strata_dir, helper_exes, python, bind) -> {unit_name: text};
            install_units(units, unit_dir, apply=False) -> [action strings];  check_health(port, ...) -> Health.
            Imports: plan's types only; build's output arrives as plain data (server_cmd, helper_exes by vendor).
            Provisional ConfigOutput{cfg_json_path: str (config writes the cfg.json the Strata server wrapper reads), server_env: dict, helper_args: {helper_index: list},
            helper_env: {helper_index: dict}, port: int} (config.py is another session's).
FAILURE   : unit rendering refuses (ServiceError) a relative executable path or a missing helper binary/args; install
            is DRY-RUN unless apply=True and never touches systemd otherwise; an unhealthy server raises ServiceError
            carrying the reason, never returns ok=False quietly.
CHECKER   : tests/install/test_services.py (unit text, dry-run default, mocked HTTP health; one check sabotaged).
Linux first; Windows is out of scope.  helpers start first: the server unit Requires= and is After= every helper.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional, Sequence

from ._build_rocm import RocmError, rocm_layout
from ._exposure import ExposureError, cleartext_warning, key_wrapper_argv, require_key_for_host
from ._http import loopback_open
from ._log import describe, scrub
from .plan import Plan

_log = logging.getLogger(__name__)

SERVER_UNIT = "seabraid-server.service"
TEST_PROMPT = "Count from 1 to 50, separated by commas, and nothing else."     # fixed on purpose: comparable runs
TEST_MAX_TOKENS = 128


class ServiceError(Exception):
    pass


def helper_unit_name(n: int) -> str:
    return f"seabraid-helper-{n}.service"


# ---- config adapter: the ONLY place the ConfigOutput's attributes are read (provisional contract) ---------------
def adapt_config(cfg: Any) -> dict:
    g = getattr
    return {"cfg_json_path": str(g(cfg, "cfg_json_path")), "server_env": dict(g(cfg, "server_env", {})),
            "helper_args": {int(k): list(v) for k, v in dict(g(cfg, "helper_args", {})).items()},
            "helper_env": {int(k): dict(v) for k, v in dict(g(cfg, "helper_env", {})).items()}, "port": int(g(cfg, "port"))}


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHM_NAME = re.compile(r"[A-Za-z0-9._-]+")


def _plain(value: Any, what: str) -> str:
    """``value`` as text, or ServiceError if it holds a newline or any other control character.  WHY refuse instead of
    escape: a newline in a unit value starts a NEW directive (``Environment=A=b`` + ``ExecStartPre=...``), and no value the
    installer builds legitimately contains one, so a control character means a poisoned input and the install must stop."""
    text = str(value)
    if _CONTROL.search(text):
        raise ServiceError(f"refusing to write a systemd unit: {what} contains a newline or control character")
    return text


def _q(arg: str) -> str:
    """systemd ExecStart quoting: '%' and '$' are specifiers/expansions there, whitespace needs quotes."""
    a = _plain(arg, "an ExecStart/Environment value").replace("%", "%%").replace("$", "$$")
    if not a or any(c in a for c in ' \t"\'\\;'):
        a = '"' + a.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return a


def _abs(path: str, what: str) -> str:
    if not PurePosixPath(path).is_absolute():
        raise ServiceError(f"{what} must be an absolute path for systemd, got {path!r}")
    return path


def _unit(desc: str, after: Sequence[str], exec_argv: Sequence[str], env: Mapping, start_timeout: int,
          pre: Optional[Sequence[str]] = None, role: str = "server") -> str:
    unit = ["[Unit]", "Description=" + _plain(desc, "the description").replace("%", "%%")]
    if after:
        unit += [f"Requires={' '.join(_plain(a, 'a unit name') for a in after)}", f"After={' '.join(after)}"]     # the server needs its helpers up first
    svc = ["[Service]", "Type=simple", *sandbox_lines(role)]       # WHY per row: see SANDBOX
    for k in env:
        if not _ENV_KEY.fullmatch(str(k)):
            raise ServiceError(f"refusing to write a systemd unit: {str(k)[:40]!r} is not an environment variable name")
    svc += [f"Environment={_q(f'{k}={v}')}" for k, v in sorted(env.items())]
    svc += (["ExecStartPre=" + " ".join(_q(a) for a in pre)] if pre else [])
    svc += ["ExecStart=" + " ".join(_q(a) for a in exec_argv), "Restart=on-failure", "RestartSec=5",
            f"TimeoutStartSec={start_timeout}"]
    return "\n".join([*unit, "", *svc, "", "[Install]", "WantedBy=default.target", ""])


# Sandbox directives as (directive, roles, WHY).  A row is data so a reviewer can read the whole policy in one place and a
# test can check every row lands in the right unit.  Nothing here hides a device node or /dev/shm (PrivateDevices, PrivateIPC,
# PrivateTmp stay out: the channel is a /dev/shm mapping shared by these processes and the GPU needs /dev/nvidia*, /dev/kfd,
# /dev/dri).  ProtectSystem=strict keeps /dev, /proc and /sys writable by design, so /dev/shm still works.
BOTH, HELPER = ("helper", "server"), ("helper",)
SANDBOX = (
    ("NoNewPrivileges=yes", BOTH, "nothing started here can gain privileges through setuid/file caps; GPU access is by device-node permissions"),
    ("RestrictSUIDSGID=yes", BOTH, "the services create no setuid/setgid files"),
    ("LockPersonality=yes", BOTH, "no process here changes its execution domain; CUDA and ROCm do not call personality(2)"),
    ("RestrictRealtime=yes", BOTH, "nothing here needs realtime scheduling; a compromised process cannot starve the host with it"),
    ("SystemCallArchitectures=native", BOTH, "blocks 32-bit syscall ABI tricks; every binary here is native"),
    ("ProtectKernelTunables=yes", BOTH, "/proc/sys and /sys are read-only to the service; it reads GPU state, never tunes the kernel"),
    ("ProtectControlGroups=yes", BOTH, "the cgroup tree is read-only to the service; it never edits cgroups"),
    ("UMask=0077", BOTH, "files and the /dev/shm channel it creates are private to this user; helper and server run as the same user"),
    ("RestrictAddressFamilies=AF_UNIX AF_NETLINK", HELPER, "helpers never touch the network; AF_UNIX and AF_NETLINK stay because CUDA/ROCm use them for driver IPC and udev"),
    ("ProtectSystem=strict", HELPER, "helpers write only /dev/shm (an API filesystem, still writable) and the CUDA cache named below, not the OS tree"),
    ("ProtectHome=read-only", HELPER, "model packs live under $HOME and are only read; read-only keeps them readable"),
    ("ReadWritePaths=-%h/.nv", HELPER, "the CUDA JIT cache lives in ~/.nv; '-' = ignore when absent"),
)


def sandbox_lines(role: str) -> list:
    """The SANDBOX directives that apply to a ``helper`` or ``server`` unit."""
    return [d for d, roles, _why in SANDBOX if role in roles]


def _flag_value(args: Sequence[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in args)


SHM_WAIT_S = 90          # helpers must have created their shm before the server starts; the engine itself retries ~60 s


def _terminal_root(env: Mapping) -> tuple:
    """(Strata's terminal dir or None, why-not text): ${XDG_DATA_HOME:-$HOME/.local/share}/strata/terminal."""
    data = env.get("XDG_DATA_HOME") or (str(PurePosixPath(env["HOME"]) / ".local" / "share") if env.get("HOME") else "")
    if not data:
        return None, "neither XDG_DATA_HOME nor HOME is set"
    root = Path(data) / "strata" / "terminal"
    return (root, "") if root.is_dir() else (None, f"no {root}")


def _version_dirs(root: Path) -> list:
    """The version directories under `root`, highest numeric version first ('0.1.56' > '0.1.9')."""
    keyed = ((tuple(int(n) for n in re.findall(r"\d+", d.name)), d) for d in root.iterdir() if d.is_dir() and re.search(r"\d", d.name))
    return [d for _, d in sorted(keyed, key=lambda t: t[0], reverse=True)]


def _best_python(version_dir: Path, executable: Callable[[str], bool]) -> Optional[str]:
    """The highest-numbered executable python3[.N] in <version_dir>/python/bin, or None; a non-executable file is never chosen."""
    pys = [p for p in (version_dir / "python" / "bin").glob("python3*") if re.fullmatch(r"python3(\.\d+)?", p.name) and executable(str(p))]
    return str(max(pys, key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)])) if pys else None


def _bundled_terminal_python(env: Mapping, executable: Callable[[str], bool]) -> tuple:
    """Strata's own interpreter in its DATA dir: <terminal dir>/<ver>/python/bin/python3[.N].  Highest numeric version wins.
    -> (path or None, why-not text)."""
    root, why = _terminal_root(env)
    if root is None:
        return None, why
    for d in _version_dirs(root):
        found = _best_python(d, executable)
        if found:
            return found, ""
    return None, f"no executable python3 under {root}/<version>/python/bin"


def pick_python(strata_dir: str, echo: Callable[[str], None] = print, executable: Callable[[str], bool] = lambda p: os.access(p, os.X_OK),
                env: Mapping = os.environ) -> str:
    """The server's interpreter, in order: (1) Strata's bundled python in its data dir (highest version),
    (2) <strata_dir>/.venv/bin/python, (3) python3 from PATH.  Prints which and why.  systemd needs an absolute path."""
    found, why = _bundled_terminal_python(env, executable)
    if found:
        echo(f"server python: Strata's bundled interpreter {found} (highest version in its data dir)")
        return found
    venv = str(PurePosixPath(strata_dir) / ".venv" / "bin" / "python")
    if executable(venv):
        echo(f"server python: {venv} (data-dir interpreter not used: {why})")
        return venv
    syspy = shutil.which("python3")
    if not syspy:
        raise ServiceError(f"no python for the server: {why}; no {venv}; no python3 on PATH")
    echo(f"server python: system python3 {syspy} ({why}; no {venv})")
    return syspy


def _visibility(vendor: str, index: int, uuid: Optional[str] = None) -> dict:
    """Mask one process to one card.  Index = the vendor tool's own ordinal (Gpu.index); PCI order keeps CUDA's
    numbering equal to nvidia-smi's."""
    if vendor == "nvidia":                  # a UUID survives re-enumeration; the index is the fallback
        return {"CUDA_VISIBLE_DEVICES": uuid or str(index), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    return {"HIP_VISIBLE_DEVICES": str(index), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


# WHY positional arguments: the shm names and the wait are DATA to a fixed script, never text spliced into it, so no name can
# change what the shell runs.
_SHM_WAIT_SCRIPT = ('w=$1; shift; i=0; while [ $i -lt $w ]; do miss=""; for n in "$@"; do [ -e "/dev/shm/$n" ] || miss="$miss $n"; done; '
                    '[ -z "$miss" ] && exit 0; i=$((i+1)); sleep 1; done; '
                    'echo "seabraid: helper shared memory still missing after ${w}s:$miss (is its helper unit running?)" >&2; exit 1')


def _shm_wait(shms: Sequence[str]) -> list:
    """ExecStartPre argv: wait until every helper's /dev/shm entry exists, fail with a message naming the missing ones."""
    names = [s.lstrip("/") for s in shms]
    for n in names:
        if not _SHM_NAME.fullmatch(n) or n in (".", ".."):
            raise ServiceError(f"shared-memory name {n[:40]!r} must be letters, digits, '.', '_' or '-' only")
    return ["/bin/sh", "-c", _SHM_WAIT_SCRIPT, "seabraid-shm-wait", str(SHM_WAIT_S), *names]


def _runtime_libs(b: Any, exists: Callable[[str], bool]) -> tuple:
    """WHY the layout probe and not b.lib_dirs: a TheRock engine needs rocm_sysdeps/lib too, and _build_rocm is its one owner."""
    if b.kind != "rocm":
        return tuple(b.lib_dirs)
    try:
        return rocm_layout(b, exists).lib_dirs
    except RocmError as e:
        raise ServiceError(str(e)) from None


def _refuse_exposed(bind: str, api_key_file: Optional[str], key_check: Callable) -> None:
    """ServiceError (and a log line) when `bind` is reachable beyond this PC without a valid key file.  `key_check` is a seam:
    tests that only render a unit need no real key file."""
    try:
        key_check(bind, api_key_file)
    except ExposureError as e:
        _log.error("refusing to render units: %s", describe(e))
        raise ServiceError(str(e)) from None
    note = cleartext_warning(bind)
    if note:                                # WHY here too: render_units is the library entry; the CLI is not the only caller
        _log.warning("bind %s: %s", bind, note)


def _server_argv(python: str, strata_dir: str, c: Mapping, bind: str, parallel: int, api_key_file: Optional[str]) -> list:
    """The server's command.  Loopback without a key keeps --no-api-key (unchanged); a key file swaps it for a start-up wrapper
    that reads the key into the environment, so the key is in no argument and no unit."""
    srv = [_abs(python, "server python"), _abs(str(PurePosixPath(strata_dir) / "serve" / "server.py"), "server.py"), "--engine", "strata",
           "--config", _abs(c["cfg_json_path"], "cfg.json"), "--host", bind, "--port", str(c["port"]),
           *([] if api_key_file else ["--no-api-key"]), "--parallel", str(parallel)]
    return key_wrapper_argv(srv, _abs(api_key_file, "--api-key-file")) if api_key_file else srv


def _helper_argv(exe: str, args: list, shm: str) -> list:
    """The helper's command: its config args minus any --device, then --shm if absent, then ``--device 0`` (always 0: see mask)."""
    kept = [a for i, a in enumerate(args) if a != "--device" and (i == 0 or args[i - 1] != "--device")]
    argv = [_abs(exe, "helper binary"), *kept]
    if not _flag_value(kept, "--shm"):                  # the helper reads flags, not SEABRAID_STRATA_SHM
        argv += ["--shm", shm]
    return argv + ["--device", "0"]


def _helper_unit(r: Any, c: Mapping, helper_exes: Mapping[str, str], libs: Mapping[str, str]) -> tuple:
    """(unit name, unit text) for one helper row of the plan."""
    exe = helper_exes.get(r.vendor)
    if not exe:
        raise ServiceError(f"no seabraid-helper binary for vendor {r.vendor!r} (helper {r.helper_index})")
    args = c["helper_args"].get(r.helper_index)
    if args is None:
        raise ServiceError(f"config has no helper args for helper {r.helper_index}")
    env = {"SEABRAID_STRATA_SHM": r.shm, **_visibility(r.vendor, r.gpu_index, r.uuid), **({"LD_LIBRARY_PATH": libs[r.vendor]} if r.vendor in libs else {}),
           **c["helper_env"].get(r.helper_index, {})}
    text = _unit(f"seabraid helper {r.helper_index} ({r.vendor} gpu {r.gpu_index}, {r.name})", [], _helper_argv(exe, args, r.shm), env, 300, role="helper")
    return helper_unit_name(r.helper_index), text


def _server_env(plan: Plan, c: Mapping, libs: Mapping[str, str]) -> dict:
    """The server's env: its card's mask, its vendor's libs, then config's env last.  WHY nothing more: config owns SEABRAID_STRATA_SHM and the
    fixed switches (the first-channel-only bug lived in a second copy), so this renders them and refuses a plan with helpers but no list."""
    main = plan.main()
    env = _visibility(main.vendor, main.gpu_index, main.uuid)
    if plan.helpers() and not c["server_env"].get("SEABRAID_STRATA_SHM"):
        raise ServiceError("config's server_env has no SEABRAID_STRATA_SHM, so the server could not find its %d helper channel(s)" % len(plan.helpers()))
    if main.vendor in libs:
        env["LD_LIBRARY_PATH"] = libs[main.vendor]
    env.update(c["server_env"])
    return env


def render_units(plan: Plan, cfg: Any, strata_dir: str, helper_exes: Mapping[str, str], python: Optional[str] = None,
                 bind: str = "127.0.0.1", parallel: int = 1,
                 exists: Callable[[str], bool] = os.path.exists, api_key_file: Optional[str] = None,
                 key_check: Callable[[str, Optional[str]], None] = require_key_for_host) -> dict:
    """One unit per helper plus the server unit.  Helpers: SEABRAID_STRATA_SHM=/seabraidN, a visibility mask for their own
    card and `--device 0` inside it.  Server: Strata's serve/server.py wrapper reading config's cfg.json, with the
    helper env, and an ExecStartPre that waits for every helper's shm.  A bind address beyond loopback is refused unless
    ``api_key_file`` names a valid key file (see _exposure); with one, the server starts with that key instead of --no-api-key."""
    _refuse_exposed(bind, api_key_file, key_check)
    c = adapt_config(cfg)
    if not plan.ready:
        raise ServiceError("plan is not ready; refusing to write units:\n" + plan.explain())
    python = python or pick_python(strata_dir)
    libs = {b.vendor: ":".join(_runtime_libs(b, exists)) for b in plan.builds}
    libs = {v: l for v, l in libs.items() if l}
    units = dict(_helper_unit(r, c, helper_exes, libs) for r in plan.helpers())
    srv = _server_argv(python, strata_dir, c, bind, parallel, api_key_file)
    pre = _shm_wait([p.shm for p in plan.helpers()]) if plan.helpers() else None
    units[SERVER_UNIT] = _unit("seabraid server (Strata main engine)", list(units), srv, _server_env(plan, c, libs), 900, pre)   # model load takes minutes
    return units


def _run(argv: list) -> int:
    rc = subprocess.run(argv).returncode
    if rc:
        _log.error("%s exited %d", " ".join(argv[:3]), rc)    # systemctl --user <verb>: no paths or values in the first three words
    return rc


def install_units(units: Mapping[str, str], unit_dir: str, apply: bool = False, run: Callable[[list], int] = _run,
                  echo: Callable[[str], None] = print, start: bool = True) -> list:
    """Write the units, `systemctl --user daemon-reload`, `enable --now` helpers first.  apply=False (the default)
    only reports what it would do; nothing is written and systemd is not touched.  start=False (--no-start) still writes
    the files and reloads systemd so `systemctl --user start` finds them, but enables and starts nothing."""
    order = sorted(units, key=lambda n: n == SERVER_UNIT)          # helpers, then the server
    acts = [f"write {Path(unit_dir) / n}" for n in order]
    cmds = [["systemctl", "--user", "daemon-reload"]] + ([["systemctl", "--user", "enable", "--now", *order]] if start else [])
    acts += [" ".join(c) for c in cmds]
    if not apply:
        for a in acts:
            echo("[dry-run] " + a)
        return acts
    Path(unit_dir).mkdir(parents=True, exist_ok=True)
    for n in order:
        (Path(unit_dir) / n).write_text(units[n], encoding="utf-8")
    for c in cmds:
        if run(c):
            raise ServiceError(f"command failed: {' '.join(c)}")
    return acts


# ---- health ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Health:
    ok: bool
    tokens: int
    decode_tok_s: Optional[float]       # None = the response carried no timings (never estimated from wall clock)
    detail: str


def _http(method: str, url: str, body: Optional[dict], timeout: float, api_key: Optional[str] = None) -> tuple:
    headers = {"Content-Type": "application/json", **({"Authorization": "Bearer " + api_key} if api_key else {})}
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(), headers=headers)
    with loopback_open(req, timeout) as r:        # never through a proxy (see _http.py)
        raw = r.read()
        return r.status, (json.loads(raw) if raw.strip().startswith(b"{") else {})


def parse_decode_tok_s(resp: Mapping) -> Optional[float]:
    """Decode speed from the response's own timings: predicted_per_second, else predicted_n over predicted_ms."""
    t = resp.get("timings") or {}
    if t.get("predicted_per_second"):
        return float(t["predicted_per_second"])
    if t.get("predicted_n") and t.get("predicted_ms"):
        return float(t["predicted_n"]) / (float(t["predicted_ms"]) / 1000.0)
    return None


def verify_shm(plan: Plan, stat: Callable[[str], Any] = os.stat, echo: Callable[[str], None] = print) -> None:
    """Each helper's live /dev/shm segment must be exactly plan.shm_bytes_per_helper.  A difference means our mirror of the
    patch's Layout::of has drifted from the patch: fail loudly with both numbers.  A missing file is skipped with a note."""
    for r in plan.helpers():
        path = "/dev/shm/" + r.shm.lstrip("/")
        try:
            size = stat(path).st_size
        except FileNotFoundError:
            echo(f"shm check skipped: {path} does not exist")
            continue
        if size != plan.shm_bytes_per_helper:
            raise ServiceError(f"{path} is {size:,} B but the plan expected {plan.shm_bytes_per_helper:,} B: the installer's "
                               "channel layout (_plan_shm.channel_bytes) has drifted from patches/strata/0001 Layout::of")
        echo(f"shm check: {path} = {size:,} B, matches the plan")


def _with_key(http: Callable, api_key: Optional[str]) -> Callable:
    """`http` that sends the key.  Only the default transport takes one; an injected transport (tests) keeps its 4-argument form."""
    return functools.partial(http, api_key=api_key) if api_key else http


def _wait_healthy(http: Callable, base: str, wait_s: float, sleep: Callable[[float], None], clock: Callable[[], float]) -> None:
    """Poll /health every 2 s until 200; ServiceError after wait_s."""
    deadline, last = clock() + wait_s, "no response"
    while True:
        try:
            status, _ = http("GET", base + "/health", None, 5.0)
            if status == 200:
                return
            last = f"/health returned {status}"
        except Exception as e:                                  # noqa: BLE001 (any failure = not up yet)
            last = f"{type(e).__name__}: {e}"
            _log.debug("health poll: %s", describe(e))
        if clock() >= deadline:
            _log.error("server not healthy after %.0f s: %s", wait_s, scrub(last))
            raise ServiceError(f"server on {base} not healthy after {wait_s:.0f}s: {last}")
        sleep(2.0)


def _reply(resp: Mapping) -> tuple:
    """(reply text, completion tokens) of a chat response; ("", 0) for whatever is missing."""
    text = (((resp.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
    return text, int((resp.get("usage") or {}).get("completion_tokens") or 0)


def _test_chat(http: Callable, base: str, model: Optional[str]) -> tuple:
    """Send the fixed test chat; -> (response dict, completion tokens).  ServiceError when it fails or returns nothing."""
    body = {"messages": [{"role": "user", "content": TEST_PROMPT}], "max_tokens": TEST_MAX_TOKENS, "temperature": 0, "stream": False}
    if model:
        body["model"] = model
    try:
        status, resp = http("POST", base + "/v1/chat/completions", body, 300.0)
    except Exception as e:                                      # noqa: BLE001
        _log.error("test chat request failed: %s", describe(e))
        raise ServiceError(f"test chat request failed: {type(e).__name__}: {e}") from e
    text, tokens = _reply(resp)
    if status != 200 or not text.strip() or tokens <= 0:
        raise ServiceError(f"test chat unhealthy: status {status}, {tokens} tokens, reply {text[:80]!r}")
    return resp, tokens


def check_health(port: int, host: str = "127.0.0.1", model: Optional[str] = None, wait_s: float = 900.0,
                 http: Callable[..., tuple] = _http, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, echo: Callable[[str], None] = print,
                 plan: Optional[Plan] = None, stat: Callable[[str], Any] = os.stat, api_key: Optional[str] = None) -> Health:
    """Poll /health until 200 (up to wait_s), send the fixed test chat, report decode tok/s.  Raises ServiceError
    when the server never becomes healthy, a helper's live shm size differs from `plan`, or the chat fails or
    returns nothing."""
    http = _with_key(http, api_key)
    base = f"http://{host}:{port}"
    _wait_healthy(http, base, wait_s, sleep, clock)
    if plan is not None:
        verify_shm(plan, stat, echo)
    resp, tokens = _test_chat(http, base, model)
    tps = parse_decode_tok_s(resp)
    detail = f"decode {tps:.1f} tok/s over {tokens} tokens" if tps else f"{tokens} tokens; response has NO timings, tok/s not measured"
    echo("health: ok, " + detail)
    return Health(True, tokens, tps, detail)
