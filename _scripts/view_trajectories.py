#!/usr/bin/env python
"""Local, zero-install web viewer for browsing proteinzen sample/trajectory PDBs.

Serves a file-browser + embedded 3D viewer (NGL.js, loaded from CDN) over the
directory you point it at -- works for both `sample.py`'s output
(`<out_dir>/samples/*.pdb`, `<out_dir>/traj/*_clean_traj.pdb` /
`*_prot_traj.pdb`) and training's `run_epoch_sample` output
(`<log_dir>/epoch_samples/epoch_NNNN/<split>/*.pdb`, including
`*_traj_noise.pdb` / `*_traj_clean.pdb`). Both write standard multi-MODEL PDBs
for trajectories, which NGL plays back natively with a frame slider -- no
conversion needed.

Usage:
    python _scripts/view_trajectories.py --dir /path/to/outputs [--port 8765]

Then open http://localhost:8765 in a browser. For a remote CoreHPC directory,
run this ON the login/compute node and tunnel the port instead of copying
files down:
    ssh -L 8765:localhost:8765 chpc-ucsf-login
    # then, on that ssh session:
    python _scripts/view_trajectories.py --dir /mnt/scratch/.../sampling/pallatom/...

Read-only: only serves files ending in .pdb/.json under --dir, nothing else.
"""
import argparse
import html
import json
import os
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = None  # set in main()
MAX_SEARCH_RESULTS = 500

PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>proteinzen trajectory viewer</title>
<script src="https://cdn.jsdelivr.net/npm/ngl@2.2.1/dist/ngl.js"></script>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #1e1e1e; color: #ddd; height: 100vh; display: flex; overflow: hidden; }
  #sidebar { width: 380px; min-width: 260px; border-right: 1px solid #333; display: flex;
             flex-direction: column; height: 100vh; }
  #search-box { padding: 8px; border-bottom: 1px solid #333; }
  #search-box input { width: 100%; padding: 6px 8px; background: #2a2a2a; color: #ddd;
                       border: 1px solid #444; border-radius: 4px; font-size: 13px; }
  #breadcrumb { padding: 6px 8px; font-size: 12px; color: #999; border-bottom: 1px solid #333;
                white-space: nowrap; overflow-x: auto; }
  #breadcrumb a { color: #7ab7ff; text-decoration: none; cursor: pointer; }
  #breadcrumb a:hover { text-decoration: underline; }
  #listing { flex: 1; overflow-y: auto; }
  .entry { padding: 5px 10px; cursor: pointer; font-size: 13px; white-space: nowrap;
           overflow: hidden; text-overflow: ellipsis; border-bottom: 1px solid #262626; }
  .entry:hover { background: #2a2a2a; }
  .entry.dir { color: #e8c46a; }
  .entry.dir::before { content: "\\1F4C1  "; }
  .entry.pdb::before { content: "\\1F9EC  "; }
  .entry.traj::before { content: "\\1F3AC  "; }
  .entry.active { background: #33475b; }
  #main { flex: 1; display: flex; flex-direction: column; height: 100vh; }
  #viewport { flex: 1; position: relative; }
  #controls { padding: 8px 12px; border-top: 1px solid #333; background: #181818;
              display: flex; align-items: center; gap: 10px; font-size: 13px; }
  #controls button { background: #333; color: #ddd; border: 1px solid #555; border-radius: 4px;
                      padding: 4px 10px; cursor: pointer; }
  #controls button:hover { background: #444; }
  #frame-slider { flex: 1; }
  #filename { padding: 6px 12px; font-size: 12px; color: #999; border-top: 1px solid #333;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #empty-msg { display: flex; align-items: center; justify-content: center; height: 100%;
               color: #666; font-size: 14px; }
</style>
</head>
<body>
  <div id="sidebar">
    <div id="search-box"><input id="search" placeholder="search filenames (recursive)..."></div>
    <div id="breadcrumb"></div>
    <div id="listing"></div>
  </div>
  <div id="main">
    <div id="viewport"><div id="empty-msg">click a .pdb file to view it</div></div>
    <div id="controls" style="display:none">
      <button id="play-btn">&#9654;</button>
      <input type="range" id="frame-slider" min="0" max="0" value="0">
      <span id="frame-label">0 / 0</span>
      <button id="reset-view-btn">reset view</button>
    </div>
    <div id="filename"></div>
  </div>

<script>
var stage = null;
var currentComp = null;
var currentTraj = null;
var playing = false;
var playTimer = null;
var curPath = "";

function initStage() {
  stage = new NGL.Stage("viewport", {backgroundColor: "#1e1e1e"});
  window.addEventListener("resize", function () { stage.handleResize(); });
}

function esc(s) { var d = document.createElement("div"); d.innerText = s; return d.innerHTML; }

function renderBreadcrumb(path) {
  var parts = path.split("/").filter(Boolean);
  var bc = document.getElementById("breadcrumb");
  var acc = "";
  var out = '<a onclick="listDir(\\'\\')">root</a>';
  for (var i = 0; i < parts.length; i++) {
    acc += (acc ? "/" : "") + parts[i];
    out += ' / <a onclick="listDir(\\'' + esc(acc) + '\\')">' + esc(parts[i]) + '</a>';
  }
  bc.innerHTML = out;
}

function classify(name) {
  if (/_traj|_clean_traj|_prot_traj/.test(name)) return "traj";
  return "pdb";
}

function renderListing(entries, activePath) {
  var el = document.getElementById("listing");
  el.innerHTML = "";
  entries.dirs.forEach(function (d) {
    var div = document.createElement("div");
    div.className = "entry dir";
    div.innerText = d;
    div.onclick = function () { listDir(curPath ? curPath + "/" + d : d); };
    el.appendChild(div);
  });
  entries.files.forEach(function (f) {
    var div = document.createElement("div");
    var cls = classify(f);
    div.className = "entry " + cls;
    div.innerText = f;
    var full = curPath ? curPath + "/" + f : f;
    if (full === activePath) div.classList.add("active");
    div.onclick = function () { loadFile(full); };
    el.appendChild(div);
  });
  if (!entries.dirs.length && !entries.files.length) {
    el.innerHTML = '<div style="padding:12px;color:#666;font-size:12px">empty</div>';
  }
}

function listDir(path) {
  curPath = path;
  renderBreadcrumb(path);
  fetch("/api/list?path=" + encodeURIComponent(path))
    .then(function (r) { return r.json(); })
    .then(function (data) { renderListing(data, null); });
}

function doSearch(q) {
  if (!q) { listDir(curPath); return; }
  fetch("/api/search?q=" + encodeURIComponent(q))
    .then(function (r) { return r.json(); })
    .then(function (data) {
      var el = document.getElementById("listing");
      el.innerHTML = "";
      document.getElementById("breadcrumb").innerHTML =
        '<span style="color:#7ab7ff">search results for "' + esc(q) + '" (' + data.results.length + (data.truncated ? "+" : "") + ")</span>";
      data.results.forEach(function (full) {
        var div = document.createElement("div");
        div.className = "entry " + classify(full);
        div.innerText = full;
        div.onclick = function () { loadFile(full); };
        el.appendChild(div);
      });
    });
}

var searchTimer = null;
document.getElementById("search").addEventListener("input", function (e) {
  clearTimeout(searchTimer);
  var v = e.target.value.trim();
  searchTimer = setTimeout(function () { doSearch(v); }, 250);
});

function stopPlaying() {
  playing = false;
  document.getElementById("play-btn").innerHTML = "&#9654;";
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
}

function setFrame(i) {
  if (!currentTraj) return;
  currentTraj.setFrame(i);
  document.getElementById("frame-slider").value = i;
  document.getElementById("frame-label").innerText = i + " / " + (currentTraj.frameCount - 1);
}

function loadFile(path) {
  document.querySelectorAll(".entry").forEach(function (e) { e.classList.remove("active"); });
  document.getElementById("filename").innerText = path;
  stopPlaying();
  if (currentComp) { stage.removeComponent(currentComp); currentComp = null; currentTraj = null; }

  fetch("/api/file?path=" + encodeURIComponent(path))
    .then(function (r) { return r.text(); })
    .then(function (text) {
      var blob = new Blob([text], {type: "text/plain"});
      stage.loadFile(blob, {ext: "pdb", name: path}).then(function (comp) {
        currentComp = comp;
        comp.addRepresentation("cartoon", {color: "chainindex"});
        comp.addRepresentation("ball+stick", {sele: "hetero and not water"});
        comp.addRepresentation("licorice", {sele: "sidechainAttached"});
        comp.autoView();

        var frameCount = (comp.structure && comp.structure.frames) ? comp.structure.frames.length : 1;
        var controls = document.getElementById("controls");
        if (frameCount > 1) {
          currentTraj = comp.addTrajectory().trajectory;
          controls.style.display = "flex";
          var slider = document.getElementById("frame-slider");
          slider.max = frameCount - 1;
          slider.value = 0;
          document.getElementById("frame-label").innerText = "0 / " + (frameCount - 1);
        } else {
          controls.style.display = "none";
        }
      });
    });
}

document.getElementById("frame-slider").addEventListener("input", function (e) {
  setFrame(parseInt(e.target.value, 10));
});

document.getElementById("play-btn").addEventListener("click", function () {
  if (!currentTraj) return;
  if (playing) { stopPlaying(); return; }
  playing = true;
  document.getElementById("play-btn").innerHTML = "&#10074;&#10074;";
  var slider = document.getElementById("frame-slider");
  playTimer = setInterval(function () {
    var next = (parseInt(slider.value, 10) + 1) % (currentTraj.frameCount);
    setFrame(next);
  }, 120);
});

document.getElementById("reset-view-btn").addEventListener("click", function () {
  if (currentComp) currentComp.autoView();
});

initStage();
listDir("");
</script>
</body>
</html>
"""


def _safe_join(rel_path: str) -> str:
    rel_path = rel_path.lstrip("/")
    full = os.path.normpath(os.path.join(ROOT, rel_path))
    if not (full == ROOT or full.startswith(ROOT + os.sep)):
        raise ValueError("path escapes root")
    return full


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep stdout quiet; this is a local dev tool

    def _send(self, code, content_type, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        try:
            if parsed.path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))

            elif parsed.path == "/api/list":
                rel = qs.get("path", [""])[0]
                full = _safe_join(rel)
                dirs, files = [], []
                for entry in sorted(os.listdir(full)):
                    if entry.startswith("."):
                        continue
                    p = os.path.join(full, entry)
                    if os.path.isdir(p):
                        dirs.append(entry)
                    elif entry.endswith(".pdb"):
                        files.append(entry)
                self._send(200, "application/json", json.dumps({"dirs": dirs, "files": files}).encode())

            elif parsed.path == "/api/search":
                q = qs.get("q", [""])[0].lower()
                results = []
                truncated = False
                for dirpath, dirnames, filenames in os.walk(ROOT):
                    dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                    for f in filenames:
                        if f.endswith(".pdb") and q in f.lower():
                            rel = os.path.relpath(os.path.join(dirpath, f), ROOT)
                            results.append(rel.replace(os.sep, "/"))
                            if len(results) >= MAX_SEARCH_RESULTS:
                                truncated = True
                                break
                    if truncated:
                        break
                self._send(200, "application/json", json.dumps({"results": results, "truncated": truncated}).encode())

            elif parsed.path == "/api/file":
                rel = qs.get("path", [""])[0]
                if not rel.endswith(".pdb"):
                    self._send(403, "text/plain", b"only .pdb files are served")
                    return
                full = _safe_join(rel)
                with open(full, "rb") as fh:
                    self._send(200, "text/plain; charset=utf-8", fh.read())

            else:
                self._send(404, "text/plain", b"not found")
        except (ValueError, FileNotFoundError, NotADirectoryError) as e:
            self._send(400, "text/plain", str(e).encode())


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="Root directory to browse (samples/traj/epoch_samples/...).")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    ROOT = os.path.abspath(args.dir)
    if not os.path.isdir(ROOT):
        raise SystemExit(f"not a directory: {ROOT}")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Serving {ROOT}")
    print(f"Open http://localhost:{args.port}")
    print("(if this is a remote/CoreHPC session, tunnel with: "
          f"ssh -L {args.port}:localhost:{args.port} <host>)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
