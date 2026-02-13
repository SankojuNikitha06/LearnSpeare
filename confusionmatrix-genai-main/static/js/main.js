function getApiKey() {
  try {
    return localStorage.getItem("learnsphere_gemini_api_key") || "";
  } catch (e) {
    return "";
  }
}

function setApiKey(key) {
  try {
    const val = key || "";
    localStorage.setItem("learnsphere_gemini_api_key", val);
    return localStorage.getItem("learnsphere_gemini_api_key") === val;
  } catch (e) {
    console.error("setApiKey failed:", e);
    return false;
  }
}

function showToast(msg, kind="info") {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = msg;
  el.classList.toggle("error", kind === "error");
  el.style.display = "block";
  clearTimeout(window.__toastT);
  window.__toastT = setTimeout(() => { el.style.display = "none"; }, 2400);
}

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body || {})
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || !data.ok) {
    const err = data.error || `Request failed: ${resp.status}`;
    throw new Error(err);
  }
  return data;
}

function escapeHtml(s) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" };
  return (s || "").replace(/[&<>"']/g, (c) => map[c]);
}

(function initNavToggle() {
  const navbar = document.getElementById("navbar");
  const toggle = document.getElementById("navToggle");
  const navlinks = document.getElementById("navlinks");
  if (!navbar || !toggle || !navlinks) return;
  toggle.addEventListener("click", function () {
    const open = navbar.classList.toggle("is-open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  });
  navlinks.querySelectorAll("a").forEach(function (a) {
    a.addEventListener("click", function () {
      navbar.classList.remove("is-open");
      toggle.setAttribute("aria-expanded", "false");
      toggle.setAttribute("aria-label", "Open menu");
    });
  });
})();
