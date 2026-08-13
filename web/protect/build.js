#!/usr/bin/env node
// Builds docs/ (the GitHub Pages source) from the plaintext site sources in
// web/ (web/index.html, web/word_associations/index.html,
// web/bibliography/index.html), replacing all three page shells with
// password-gated versions. Everything else (data.js, style.css, images,
// ...) is copied through unchanged. docs/ mirrors web/'s layout flatly
// (docs/index.html, docs/word_associations/, docs/bibliography/) and lives
// at the repo root, one level up from web/, since that's where GitHub Pages
// needs it to serve from.
//
// Crypto: AES-256-GCM, key derived from the password with PBKDF2-SHA256.
// Encryption happens here in Node (crypto.createCipheriv); decryption
// happens in the browser via the native SubtleCrypto API -- no third-party
// JS libraries, matching the rest of this site. A fresh random salt + IV is
// generated per page per build.

"use strict";

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const SITE_SRC = path.join(__dirname, ".."); // web/
const REPO_ROOT = path.join(SITE_SRC, ".."); // repo root -- docs/ lives here, alongside web/
const DOCS = path.join(REPO_ROOT, "docs");
const PASSWORD_FILE = path.join(__dirname, ".password");
const ITERATIONS = 310000;

const SKIP_NAMES = new Set([".DS_Store", "covers_cache.json"]);
const SKIP_EXT = new Set([".py", ".md"]);

const PAGES = [
  { src: path.join(SITE_SRC, "index.html"), out: path.join(DOCS, "index.html"), title: "Green Counting" },
  {
    src: path.join(SITE_SRC, "word_associations", "index.html"),
    out: path.join(DOCS, "word_associations", "index.html"),
    title: "Verbs × Keywords in Architecture Syllabi",
  },
  {
    src: path.join(SITE_SRC, "bibliography", "index.html"),
    out: path.join(DOCS, "bibliography", "index.html"),
    title: "Syllabus Citation Network",
  },
];

function readPassword() {
  if (!fs.existsSync(PASSWORD_FILE)) {
    console.error(
      "\nNo password file found at web/protect/.password\n\n" +
        "Create it yourself (outside of any AI chat, so the password never\n" +
        "ends up in a transcript) with a single line containing the shared\n" +
        "site password, e.g. from a terminal:\n\n" +
        "  echo -n 'your-password-here' > web/protect/.password\n\n" +
        "That file is already in .gitignore and will never be committed.\n" +
        "Then re-run: node web/protect/build.js\n"
    );
    process.exit(1);
  }
  const pw = fs.readFileSync(PASSWORD_FILE, "utf8").trim();
  if (!pw) {
    console.error("web/protect/.password is empty -- put the password on one line.");
    process.exit(1);
  }
  return pw;
}

function encrypt(plaintext, password) {
  const salt = crypto.randomBytes(16);
  const iv = crypto.randomBytes(12);
  const key = crypto.pbkdf2Sync(password, salt, ITERATIONS, 32, "sha256");
  const cipher = crypto.createCipheriv("aes-256-gcm", key, iv);
  const ciphertext = Buffer.concat([cipher.update(plaintext, "utf8"), cipher.final()]);
  const authTag = cipher.getAuthTag();
  // WebCrypto's AES-GCM decrypt expects ciphertext with the auth tag
  // appended at the end, so concatenate them the same way here.
  return {
    salt: salt.toString("base64"),
    iv: iv.toString("base64"),
    data: Buffer.concat([ciphertext, authTag]).toString("base64"),
  };
}

function copyTree(srcDir, outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    if (SKIP_NAMES.has(entry.name)) continue;
    const srcPath = path.join(srcDir, entry.name);
    const outPath = path.join(outDir, entry.name);
    if (entry.isDirectory()) {
      copyTree(srcPath, outPath);
    } else {
      if (SKIP_EXT.has(path.extname(entry.name))) continue;
      fs.copyFileSync(srcPath, outPath);
    }
  }
}

// Client-side gate script, injected as a plain string (no template literals
// inside, so it nests cleanly inside this file's own template literal).
function gateScript(saltB64, ivB64, dataB64) {
  const lines = [
    '(function () {',
    '  "use strict";',
    '  var ITERATIONS = ' + ITERATIONS + ";",
    '  var SALT_B64 = "' + saltB64 + '";',
    '  var IV_B64 = "' + ivB64 + '";',
    '  var DATA_B64 = "' + dataB64 + '";',
    "",
    "  function b64ToBytes(b64) {",
    "    var bin = atob(b64);",
    "    var bytes = new Uint8Array(bin.length);",
    "    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);",
    "    return bytes;",
    "  }",
    "",
    "  function reveal(html, password) {",
    '    var withPw = html.split("__PASSWORD__").join(encodeURIComponent(password));',
    "    document.open();",
    "    document.write(withPw);",
    "    document.close();",
    "  }",
    "",
    "  function attempt(password, onFail) {",
    "    var enc = new TextEncoder();",
    "    window.crypto.subtle",
    '      .importKey("raw", enc.encode(password), "PBKDF2", false, ["deriveKey"])',
    "      .then(function (baseKey) {",
    "        return window.crypto.subtle.deriveKey(",
    '          { name: "PBKDF2", salt: b64ToBytes(SALT_B64), iterations: ITERATIONS, hash: "SHA-256" },',
    "          baseKey,",
    '          { name: "AES-GCM", length: 256 },',
    "          false,",
    '          ["decrypt"]',
    "        );",
    "      })",
    "      .then(function (key) {",
    "        return window.crypto.subtle.decrypt(",
    '          { name: "AES-GCM", iv: b64ToBytes(IV_B64) },',
    "          key,",
    "          b64ToBytes(DATA_B64)",
    "        );",
    "      })",
    "      .then(function (plainBuf) {",
    "        reveal(new TextDecoder().decode(plainBuf), password);",
    "      })",
    "      .catch(onFail);",
    "  }",
    "",
    "  function wireForm() {",
    '    var form = document.getElementById("gate-form");',
    '    var input = document.getElementById("gate-password");',
    '    var err = document.getElementById("gate-error");',
    '    var btn = document.getElementById("gate-submit");',
    '    form.addEventListener("submit", function (e) {',
    "      e.preventDefault();",
    '      err.classList.remove("show");',
    "      btn.disabled = true;",
    "      attempt(input.value, function () {",
    '        err.classList.add("show");',
    "        btn.disabled = false;",
    "        input.select();",
    "      });",
    "    });",
    "    input.focus();",
    "  }",
    "",
    "  wireForm();",
    "",
    "  // A parent frame that already unlocked can pass the password along via",
    "  // the URL hash (never sent to the server) so this page unlocks silently",
    "  // instead of asking again. Direct navigation here with no hash still",
    "  // shows the normal prompt above.",
    '  var m = /(?:^#|[#&])pw=([^&]+)/.exec(window.location.hash || "");',
    "  if (m) attempt(decodeURIComponent(m[1]), function () {});",
    "})();",
  ];
  return lines.join("\n");
}

function gateHtml(title, saltB64, ivB64, dataB64) {
  return [
    "<!doctype html>",
    '<html lang="en">',
    "  <head>",
    '    <meta charset="utf-8" />',
    '    <meta name="viewport" content="width=device-width, initial-scale=1" />',
    "    <title>" + title + "</title>",
    "    <style>",
    "      :root { color-scheme: dark; }",
    "      * { box-sizing: border-box; }",
    "      html, body {",
    "        height: 100%; margin: 0;",
    '        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;',
    "        background: #000000; color: #ffffff;",
    "        display: flex; align-items: center; justify-content: center;",
    "      }",
    "      .card {",
    "        width: min(320px, 86vw); text-align: left;",
    "      }",
    "      .card h1 { font-size: 1.1rem; font-weight: 300; margin: 0 0 18px; color: #ffffff; }",
    "      .card .row { display: flex; gap: 8px; }",
    "      .card input {",
    "        flex: 1 1 auto; min-width: 0; font: inherit; font-size: 0.95rem; padding: 10px 12px;",
    "        border-radius: 6px; border: 1px solid #ffffff; background: #000000;",
    "        color: #ffffff;",
    "      }",
    "      .card input::placeholder { color: #888888; }",
    "      .card input:focus { outline: 2px solid #ffffff; outline-offset: -1px; }",
    "      .card button {",
    "        flex: 0 0 auto; width: 42px; font: inherit; font-size: 1.15rem; padding: 10px 0;",
    "        border-radius: 6px; border: 1px solid #ffffff; background: #ffffff; color: #000000; cursor: pointer;",
    "        display: flex; align-items: center; justify-content: center; line-height: 1;",
    "      }",
    "      .card button:disabled { opacity: 0.4; cursor: default; }",
    "      .card button:hover:not(:disabled) { background: #cccccc; }",
    "      #gate-error {",
    "        color: #ffffff; font-size: 0.84rem; margin-top: 12px;",
    "        visibility: hidden;",
    "      }",
    "      #gate-error.show { visibility: visible; }",
    "    </style>",
    "  </head>",
    "  <body>",
    '    <div class="card">',
    "      <h1>" + title + "</h1>",
    '      <form id="gate-form" autocomplete="off">',
    '        <div class="row">',
    '          <input id="gate-password" type="password" placeholder="Password" autocomplete="off" />',
    '          <button id="gate-submit" type="submit" aria-label="Unlock">&rarr;</button>',
    "        </div>",
    '        <p id="gate-error">Incorrect password.</p>',
    "      </form>",
    "    </div>",
    "    <script>",
    gateScript(saltB64, ivB64, dataB64),
    "    </script>",
    "  </body>",
    "</html>",
    "",
  ].join("\n");
}

function main() {
  const password = readPassword();

  fs.rmSync(DOCS, { recursive: true, force: true });
  copyTree(path.join(SITE_SRC, "word_associations"), path.join(DOCS, "word_associations"));
  copyTree(path.join(SITE_SRC, "bibliography"), path.join(DOCS, "bibliography"));

  for (const page of PAGES) {
    const plaintext = fs.readFileSync(page.src, "utf8");
    const { salt, iv, data } = encrypt(plaintext, password);
    fs.mkdirSync(path.dirname(page.out), { recursive: true });
    fs.writeFileSync(page.out, gateHtml(page.title, salt, iv, data));
    console.log("gated:", path.relative(REPO_ROOT, page.out));
  }

  console.log("\ndocs/ rebuilt. Commit it, then set GitHub Pages to serve from main:/docs.");
}

main();
