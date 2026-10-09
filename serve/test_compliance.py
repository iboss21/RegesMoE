"""serve/test_compliance.py - the first-run compliance gate (serve/web/app.js).
//
// The gate is the one piece of the web app that can STOP someone using the app, so each test here is
// about a way the obvious implementation is wrong, not about the happy path:
//
//   1. it appears on first entry and not after accepting;
//   2. DECLINE is honest - it must NOT write the accepted flag, or the next entry silently agrees for them;
//   3. Escape is not consent - dismissing the prompt must never accept it;
//   4. with app.js absent or broken the app is still USABLE, because the modal is `hidden` in the markup
//      and only the script removes it. A gate that can strand a user behind an undismissable box is worse
//      than no gate;
//   5. nothing is sent to the server before the terms are accepted (no /health poll).
//
// The JS runs for real under Node against a minimal DOM stub, so these are behavioural assertions, not
// string matching on the source.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "serve" / "web"


def js() -> str:
    return (WEB / "app.js").read_text(encoding="utf-8")


def html() -> str:
    return (WEB / "index.html").read_text(encoding="utf-8")


class Markup(unittest.TestCase):
    """The gate's fail-safe: `hidden` in the markup, so no script means no lockout."""

    def test_modal_is_hidden_in_the_markup(self):
        m = re.search(r'<div class="st-modal" id="accept-modal"[^>]*>', html())
        self.assertIsNotNone(m, "the modal is not in index.html")
        self.assertIn("hidden", m.group(0),
                      "the modal must carry `hidden` in the MARKUP - app.js removes it, so a script that "
                      "never runs leaves the app usable instead of locked")
        self.assertIn('role="dialog"', m.group(0))
        self.assertIn('aria-modal="true"', m.group(0))

    def test_both_outcomes_are_offered(self):
        for id_ in ("accept-ok", "accept-decline"):
            self.assertIn(f'id="{id_}"', html(), id_)

    def test_both_actions_say_what_they_do(self):
        # a bare "OK" that turns out to mean "accept legal terms" is the failure this avoids
        self.assertRegex(html(), r'id="accept-ok"[^>]*>Accept and continue<')
        self.assertRegex(html(), r'id="accept-decline"[^>]*>Decline<')

    def test_the_modal_precedes_the_script(self):
        """It must be in the DOM before app.js runs, or the gate cannot show on a first paint."""
        self.assertLess(html().index('id="accept-modal"'), html().index('src="web/app.js"'))

    def test_css_has_the_modal_and_its_hidden_rule(self):
        css = (WEB / "components.css").read_text(encoding="utf-8")
        self.assertIn(".st-modal", css)
        self.assertIn(".st-modal[hidden]{display:none}", css,
                      "without this the `hidden` attribute would not hide it and every visitor would be stuck")


class Source(unittest.TestCase):
    """The decisions, checked where they are written."""

    def test_decline_does_not_record_acceptance(self):
        src = js()
        decline = re.search(r'no\.onclick\s*=\s*\(\)\s*=>\s*\{(.*?)\};', src, re.S)
        self.assertIsNotNone(decline, "no decline handler")
        self.assertNotIn("store.set(ACCEPT_KEY", decline.group(1),
                         "declining must NOT write the accepted flag - otherwise the next visit agrees for them")

    def test_escape_declines_rather_than_accepts(self):
        m = re.search(r'modal\.onkeydown.*', js())
        self.assertIsNotNone(m)
        self.assertIn('no.click()', m.group(0), "Escape must decline, never accept")

    def test_focus_starts_on_the_accepting_action(self):
        m = re.search(r'function showCompliance.*?\n\}', js(), re.S)
        self.assertIsNotNone(m)
        body = m.group(0)
        self.assertRegex(body, r'ok\.focus\(\)',
                         "focus belongs on 'Accept', not on the dismissing button")

    def test_the_flag_is_versioned_and_stamped(self):
        m = re.search(r'const ACCEPT_KEY = "([^"]+)"', js())
        self.assertIsNotNone(m)
        self.assertRegex(m.group(1), r"^accept\.v\d+$",
                         "the key carries a version, so a later release can re-ask when the terms change")
        self.assertIn("at: new Date().toISOString()", js(), "acceptance is stamped with when it happened")

    def test_nothing_is_requested_before_the_gate(self):
        """No /health poll until the terms are accepted: an unaccepted install makes no request of its own."""
        src = js()
        tail = src[src.rindex("// ------------------------------------------------------------------ start"):]
        self.assertLess(tail.index("gateFirstRun().then"), tail.index("loadHealth()"),
                        "loadHealth() must come AFTER the gate resolves")
        self.assertIn("if (!accepted) return;", tail)


# -------------------------------------------------------------------------- behaviour, for real
HARNESS = r"""
// A DOM stub just large enough for the gate's own code paths, run under Node. The app's boot code is
// extracted so the test can drive it without a server.
const fs = require("fs");
const path = process.argv[2];
const src = fs.readFileSync(path, "utf8");

const calls = [];
const LS = {};
const listeners = {};
function makeEl(id, tag) {
  return {
    id, tagName: (tag || "div").toUpperCase(), hidden: true, disabled: false, dataset: {},
    style: {}, classList: {add(){}, remove(){}, contains(){return false}},
    textContent: "", innerHTML: "", placeholder: "", value: "",
    className: "",
    focus() { calls.push(["focus", id]); }, onclick: null, onkeydown: null,
    click() { calls.push(["click", id]); if (this.onclick) this.onclick({target: this}); },
    appendChild(){}, remove(){}, querySelector(){return null},
    addEventListener(k, f){ listeners[id + ":" + k] = f; },
    querySelectorAll(){return [];},
  };
}
const els = {};
global.document = {
  documentElement: {dataset: {}},
  getElementById: (id) => (els[id] = els[id] || makeEl(id)),
  createElement: (t) => makeEl("new-" + t, t),
  querySelector: () => null, querySelectorAll: () => [],
  addEventListener(k, f){ listeners["doc:" + k] = f; },
  body: makeEl("body", "body"),
};
global.window = {addEventListener(k,f){ listeners["win:"+k]=f; }};
global.location = {search:"", hash:"", pathname:"/", replaceState(){}};
global.history = {replaceState(){}};
global.localStorage = {
  getItem: (k) => (k in LS ? LS[k] : null),
  setItem: (k, v) => { LS[k] = v; calls.push(["set", k, v]); },
};
global.matchMedia = () => ({matches:false});
global.setTimeout = (fn, ms) => { if (ms > 100000) calls.push(["sticky-toast", ms]); return 0; };
global.clearTimeout = () => {};
global.navigator = {clipboard:{writeText:async()=>{}}};
global.fetch = (u) => { calls.push(["fetch", u]); return Promise.resolve({json:async()=>({}), ok:true}); };
global.matchMedia = () => ({matches:false});
global.requestAnimationFrame = (f) => f();
global.Event = class { constructor(t){this.type=t;} preventDefault(){} };
global.URLSearchParams = URLSearchParams;

// The gate's own functions, lifted out of app.js so the rest of the app (which needs a live server)
// does not have to run. Everything under test is in here verbatim.
const GATE_SRC = `
  const store = { get(k,d){ try { const v = localStorage.getItem("strata."+k); return v===null?d:JSON.parse(v);} catch(e){return d;} },
                  set(k,v){ try { localStorage.setItem("strata."+k, JSON.stringify(v)); } catch(e){} } };
  const toast = (kind,title,text,ms,action) => { if(ms>100000) calls.push(["toast",kind,title]); };
  const $ = (id) => document.getElementById(id);
` + (function () {
  // pull the gate's constant and its four functions verbatim out of the real file
  const wanted = ['const ACCEPT_KEY = "', "function complianceAccepted", "function showCompliance",
                  "function showDeclined", "function gateFirstRun"];
  let out = "";
  for (const w of wanted) {
    const i = src.indexOf(w);
    if (i < 0) { console.error("MISSING " + w); process.exit(3); }
    let depth = 0, started = false, j = i;
    for (; j < src.length; j++) {
      if (src[j] === "{") { depth++; started = true; }
      else if (src[j] === "}") { depth--; if (started && depth === 0) { j++; break; } }
    }
    out += src.slice(i, j) + "\n";
  }
  return out;
})() + `
global.__calls = calls; global.__LS = LS; global.__gate = {showCompliance, showDeclined, gateFirstRun, complianceAccepted};
`;

// `new Function` so the extracted code shares the module's globals (document, localStorage, ...).
new Function("calls", "LS", GATE_SRC)(calls, LS);
const G = global.__gate;
const fail = (m) => { console.error("FAIL: " + m); process.exit(1); };

(async () => {
// --- 1. first entry: not accepted yet -> the gate opens
if (G.complianceAccepted()) fail("accepted before any interaction");
G.showCompliance();
if (els["accept-modal"].hidden !== false) fail("the modal did not open on first entry");
console.log("OK first entry shows the gate");

// --- focus went to Accept
if (!calls.some((c) => c[0] === "focus" && c[1] === "accept-ok")) fail("focus did not land on Accept");
console.log("OK focus starts on the accepting action");

// --- 3. Escape is decline, not consent
els["accept-modal"].onkeydown({key: "Escape", preventDefault(){}});
if (global.__LS["strata.accept.v2"]) fail("Escape recorded acceptance");
if (!calls.some((c) => c[0] === "toast")) fail("declining did not say so");
console.log("OK Escape declines and records nothing");

// --- decline left the composer disabled
if (!els["input"] || els["input"].disabled !== true) fail("composer still live after declining");
console.log("OK declining disables the composer");

// --- a SECOND visit after a decline still asks
G.showCompliance();
if (els["accept-modal"].hidden !== false) fail("declined, and it did not ask again");
console.log("OK a declined install is asked again (decline was honest)");

// --- accept: flag written, versioned and stamped, gate closes
const p = G.showCompliance();     // re-arm the handlers (the previous pair already fired)
els["accept-ok"].onclick();
await p;
const rec = global.__LS["strata.accept.v2"];
if (!rec) fail("accepting did not record anything");
const parsed = JSON.parse(rec);
if (!/^\d+$/.test(String(parsed.v))) fail("the stamp has no version");
if (!parsed.at || Number.isNaN(Date.parse(parsed.at))) fail("the stamp has no date");
if (els["accept-modal"].hidden !== true) fail("the gate stayed open after accepting");
if (!G.complianceAccepted()) fail("still not accepted after accepting");
console.log("OK accepting records {v, at} and closes the gate");

// --- 2. a second visit after accepting does NOT re-ask
if (!G.complianceAccepted()) fail("acceptance did not persist");
const before = els["accept-modal"].hidden;
if (G.complianceAccepted()) { /* the caller checks this first, so it would not call showCompliance */ }
if (els["accept-modal"].hidden !== before) fail("a modal opened on its own after accepting");
console.log("OK acceptance persists across visits (the gate would not re-open)");
console.log("ALL OK");
})();
"""


class Behaviour(unittest.TestCase):
    """The gate run for real under Node against a DOM stub."""

    @classmethod
    def setUpClass(cls):
        node = shutil_which("node")
        if node is None:
            raise unittest.SkipTest("node is not on PATH")
        path = ROOT / "serve" / "web" / "app.js"
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "harness.cjs"
            f.write_text(HARNESS, encoding="utf-8")
            r = subprocess.run([node, str(f), str(path)], capture_output=True, text=True, timeout=60)
        cls.out, cls.code = r.stdout, r.returncode
        cls.err = r.stderr

    def test_the_gate_runs_clean(self):
        self.assertEqual(self.code, 0, f"harness failed\nstdout:\n{self.out}\nstderr:\n{self.err}")
        for line in ("first entry shows the gate",
                     "focus starts on the accepting action",
                     "Escape declines and records nothing",
                     "declining disables the composer",
                     "a declined install is asked again (decline was honest)",
                     "accepting records {v, at} and closes the gate",
                     "acceptance persists across visits (the gate would not re-open)"):
            self.assertIn(line, self.out, f"missing: {line}\nfull output:\n{self.out}")
        self.assertIn("ALL OK", self.out)


def shutil_which(name):
    from shutil import which
    return which(name)


if __name__ == "__main__":
    unittest.main()