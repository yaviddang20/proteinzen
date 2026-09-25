#!/usr/bin/env python
"""Local, zero-install web viewer for browsing proteinzen sample/trajectory PDBs.

Serves a file-browser + embedded 3D viewer (Mol*, loaded from CDN) over the
directory you point it at -- works for both `sample.py`'s output
(`<out_dir>/samples/*.pdb`, `<out_dir>/traj/*_clean_traj.pdb` /
`*_prot_traj.pdb`) and training's `run_epoch_sample` output
(`<log_dir>/epoch_samples/epoch_NNNN/<split>/*.pdb`, including
`*_traj_noise.pdb` / `*_traj_clean.pdb`). Both write standard multi-MODEL PDBs
for trajectories, which Mol* plays back natively via its own built-in
trajectory/animation controls -- no conversion needed.

Was originally built on NGL.js; switched to Mol* after NGL repeatedly needed
workarounds (asTrajectory defaults to false and silently merges MODEL blocks
instead of treating them as frames; weaker default secondary-structure
rendering; a leftover placeholder div interfering with the canvas). Mol* is
the more actively-maintained standard (what RCSB's own viewer uses), with a
real built-in trajectory UI (`viewportShowAnimation`) instead of a hand-rolled
frame slider.

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
<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/molstar@5.11.0/build/viewer/molstar.css" />
<script src="https://cdn.jsdelivr.net/npm/molstar@5.11.0/build/viewer/molstar.js"></script>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #1e1e1e; color: #ddd; height: 100vh; display: flex; overflow: hidden; }
  #sidebar { width: 380px; min-width: 260px; border-right: 1px solid #333; display: flex;
             flex-direction: column; height: 100vh; }
  #search-box { padding: 8px; border-bottom: 1px solid #333; }
  #search-box input { width: 100%; padding: 6px 8px; background: #2a2a2a; color: #ddd;
                       border: 1px solid #444; border-radius: 4px; font-size: 13px; }
  #filter-box { padding: 0 8px 8px 8px; border-bottom: 1px solid #333; display: flex;
                gap: 4px; align-items: center; }
  #filter-box input { background: #2a2a2a; color: #ddd; border: 1px solid #444;
                       border-radius: 4px; font-size: 12px; padding: 4px 6px; }
  #filter-metric { flex: 1.3; min-width: 0; }
  #filter-min, #filter-max { flex: 1; min-width: 0; }
  #filter-box span { color: #777; font-size: 11px; }
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
  #molstar-app { flex: 1; position: relative; }
  #filename { padding: 6px 12px; font-size: 12px; color: #999; border-top: 1px solid #333;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
</head>
<body>
  <div id="sidebar">
    <div id="search-box"><input id="search" placeholder="search filenames (recursive)..."></div>
    <div id="filter-box">
      <input id="filter-metric" placeholder="metric (e.g. ca_rmsd)">
      <span>&ge;</span><input id="filter-min" type="number" step="any" style="width:56px">
      <span>&le;</span><input id="filter-max" type="number" step="any" style="width:56px">
    </div>
    <div id="breadcrumb"></div>
    <div id="listing"></div>
  </div>
  <div id="main">
    <div id="molstar-app"></div>
    <div id="filename">click a .pdb file to view it</div>
  </div>

<script>
var viewer = null;
var viewerReady = molstar.Viewer.create("molstar-app", {
  layoutIsExpanded: false,
  layoutShowControls: true,
  layoutShowRemoteState: false,
  layoutShowSequence: true,
  layoutShowLog: false,
  layoutShowLeftPanel: true,
  viewportShowExpand: true,
  viewportShowSelectionMode: false,
  viewportShowAnimation: true
}).then(function (v) { viewer = v; return v; });

var curPath = "";

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

function filterParams() {
  var metric = document.getElementById("filter-metric").value.trim();
  var min = document.getElementById("filter-min").value;
  var max = document.getElementById("filter-max").value;
  if (!metric) return "";
  return "&metric=" + encodeURIComponent(metric) + "&min=" + encodeURIComponent(min) + "&max=" + encodeURIComponent(max);
}

function listDir(path) {
  curPath = path;
  renderBreadcrumb(path);
  fetch("/api/list?path=" + encodeURIComponent(path) + filterParams())
    .then(function (r) { return r.json(); })
    .then(function (data) { renderListing(data, null); });
}

function doSearch(q) {
  if (!q) { listDir(curPath); return; }
  fetch("/api/search?q=" + encodeURIComponent(q) + filterParams())
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

var filterTimer = null;
function refreshCurrentView() {
  var q = document.getElementById("search").value.trim();
  if (q) { doSearch(q); } else { listDir(curPath); }
}
["filter-metric", "filter-min", "filter-max"].forEach(function (id) {
  document.getElementById(id).addEventListener("input", function () {
    clearTimeout(filterTimer);
    filterTimer = setTimeout(refreshCurrentView, 250);
  });
});

function showError(msg) {
  console.error(msg);
  var el = document.getElementById("filename");
  el.innerText = msg;
  el.style.color = "#ff6b6b";
}

function loadFile(path) {
  document.querySelectorAll(".entry").forEach(function (e) { e.classList.remove("active"); });
  var filenameEl = document.getElementById("filename");
  filenameEl.style.color = "";
  filenameEl.innerText = "loading " + path + " ...";

  Promise.all([viewerReady, fetch("/api/file?path=" + encodeURIComponent(path))])
    .then(function (results) {
      var r = results[1];
      if (!r.ok) throw new Error("server returned " + r.status + " fetching " + path);
      return r.text();
    })
    .then(function (text) {
      if (!text || !text.trim()) throw new Error("file is empty: " + path);
      // clear() before loading -- otherwise each click stacks another structure
      // into the same scene instead of replacing the previous one.
      return viewer.plugin.clear().then(function () {
        return viewer.loadStructureFromData(text, "pdb", {dataLabel: path});
      });
    })
    .then(function () {
      filenameEl.innerText = path;
    })
    .catch(function (err) {
      showError("failed to load " + path + ": " + (err && err.message ? err.message : err));
    });
}

window.onerror = function (msg, src, line, col, err) {
  showError("JS error: " + msg + " (line " + line + ")");
  return false;
};
window.addEventListener("unhandledrejection", function (ev) {
  showError("unhandled promise rejection: " + (ev.reason && ev.reason.message ? ev.reason.message : ev.reason));
});

if (typeof molstar === "undefined") {
  showError("Mol* failed to load from CDN -- check network access to cdn.jsdelivr.net");
}
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


_METRIC_CACHE: dict = {}


def _metric_value(pdb_full_path: str, metric: str):
    """Every eval pipeline this session (OG plinder eval, Pallatom eval) writes
    per-sample metrics to per_sample/<stem>.json as a sibling of samples/ --
    i.e. .../out_dir/samples/foo.pdb has its metrics at
    .../out_dir/per_sample/foo.json. Returns None if there's no samples/
    sibling, no matching JSON, or the metric key isn't present/numeric."""
    parent = os.path.dirname(pdb_full_path)
    if os.path.basename(parent) != "samples":
        return None
    stem = os.path.splitext(os.path.basename(pdb_full_path))[0]
    json_path = os.path.join(os.path.dirname(parent), "per_sample", stem + ".json")

    cached = _METRIC_CACHE.get(json_path)
    if cached is None:
        if not os.path.isfile(json_path):
            return None
        try:
            with open(json_path) as f:
                cached = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        _METRIC_CACHE[json_path] = cached

    val = cached.get(metric)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    if val != val:  # NaN
        return None
    return val


def _passes_filter(pdb_full_path: str, metric: str, lo, hi) -> bool:
    if metric is None:
        return True
    val = _metric_value(pdb_full_path, metric)
    if val is None:
        return False
    if lo is not None and val < lo:
        return False
    if hi is not None and val > hi:
        return False
    return True


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
                metric = qs.get("metric", [None])[0] or None
                lo = float(qs["min"][0]) if qs.get("min") and qs["min"][0] != "" else None
                hi = float(qs["max"][0]) if qs.get("max") and qs["max"][0] != "" else None
                full = _safe_join(rel)
                dirs, files = [], []
                for entry in sorted(os.listdir(full)):
                    if entry.startswith("."):
                        continue
                    p = os.path.join(full, entry)
                    if os.path.isdir(p):
                        dirs.append(entry)
                    elif entry.endswith(".pdb") and _passes_filter(p, metric, lo, hi):
                        files.append(entry)
                self._send(200, "application/json", json.dumps({"dirs": dirs, "files": files}).encode())

            elif parsed.path == "/api/search":
                q = qs.get("q", [""])[0].lower()
                metric = qs.get("metric", [None])[0] or None
                lo = float(qs["min"][0]) if qs.get("min") and qs["min"][0] != "" else None
                hi = float(qs["max"][0]) if qs.get("max") and qs["max"][0] != "" else None
                results = []
                truncated = False
                for dirpath, dirnames, filenames in os.walk(ROOT):
                    dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                    for f in filenames:
                        if f.endswith(".pdb") and q in f.lower():
                            full_f = os.path.join(dirpath, f)
                            if not _passes_filter(full_f, metric, lo, hi):
                                continue
                            rel = os.path.relpath(full_f, ROOT)
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
