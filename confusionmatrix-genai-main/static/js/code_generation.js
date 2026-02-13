document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("codeForm");
  const topic = document.getElementById("topic");
  const depth = document.getElementById("depth");
  const btn = document.getElementById("generateBtn");
  const codeEl = document.getElementById("code");
  const depsEl = document.getElementById("deps");
  const runEl = document.getElementById("run");
  const dl = document.getElementById("downloadLink");
  const status = document.getElementById("status");
  const copyBtn = document.getElementById("copyBtn");
  const codeMessage = document.getElementById("codeMessage");

  function applyDoneState(data, lang) {
    codeEl.textContent = data.code || "";
    const badge = document.getElementById("appliedBadge");
    if (badge) {
      const raw = data.level || (typeof getProfile === "function" ? getProfile().level : "intermediate");
      const level = raw ? raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase() : "Intermediate";
      badge.innerHTML = "Personalized for: " + level;
      badge.style.display = "block";
    }
    const codeHeading = document.getElementById("codeHeading");
    if (data.is_message) {
      if (codeHeading) codeHeading.textContent = "Message";
      status.textContent = "That wasn't a coding question—no worries!";
      depsEl.textContent = "—";
      runEl.textContent = "";
      dl.style.display = "none";
    } else {
      if (codeHeading) codeHeading.textContent = (data.language || lang) + " Code";
      depsEl.textContent = (data.dependencies && data.dependencies.length)
        ? data.dependencies.join(" ")
        : (lang === "python" ? "No extra deps detected." : "—");
      runEl.textContent = data.run_instructions || "";
      if (data.download_url) {
        dl.href = data.download_url;
        dl.textContent = "Download " + (data.filename || "");
        dl.style.display = "inline-flex";
      }
      status.textContent = "Code generated. Download and run locally.";
    }
  }

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const topicVal = (topic?.value || "").trim();
    const lang = document.getElementById("language")?.value || "python";
    if (!topicVal) {
      if (codeMessage) {
        codeMessage.innerHTML = "<p>Please enter a topic (e.g. logistic regression in Python, decision tree with sklearn).</p>";
        codeMessage.classList.remove("is-error");
        codeMessage.hidden = false;
      }
      topic?.focus();
      if (typeof showToast === "function") showToast("Enter a topic first.");
      return;
    }
    btn.disabled = true;
    btn.textContent = "Generating...";
    status.textContent = "";
    codeEl.textContent = "";
    if (codeMessage) { codeMessage.innerHTML = ""; codeMessage.hidden = true; }
    const badge = document.getElementById("appliedBadge");
    if (badge) badge.style.display = "none";
    depsEl.textContent = "";
    runEl.textContent = "";
    dl.href = "#";
    dl.style.display = "none";

    try {
      const res = await fetch("/api/generate-code-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: topicVal,
          depth: depth.value,
          language: lang,
          api_key: getApiKey(),
          profile: typeof getProfile === "function" ? getProfile() : { level: "intermediate", language: "en" },
          user_id: typeof getUserId === "function" ? getUserId() : ""
        })
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || res.statusText);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let accumulated = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const line of parts) {
          if (!line.startsWith("data: ")) continue;
          let ev;
          try {
            ev = JSON.parse(line.slice(6));
          } catch (_) { continue; }
          if (ev.type === "message_only") {
            const msg = (ev.message && ev.message.trim()) || "LearnSphere is for ML code—try \"logistic regression in Python\", \"decision tree with sklearn\", or \"k-means clustering\".";
            if (codeMessage) {
              codeMessage.classList.remove("is-error");
              codeMessage.innerHTML = "<p>" + escapeHtml(msg) + "</p>";
              codeMessage.hidden = false;
            }
            return;
          }
          if (ev.type === "chunk" && ev.text) {
            accumulated += ev.text;
            codeEl.textContent = accumulated;
          }
          if (ev.type === "error") {
            if (codeMessage) {
              codeMessage.classList.add("is-error");
              codeMessage.innerHTML = "<p>" + escapeHtml(ev.error || "Error") + "</p>";
              codeMessage.hidden = false;
            }
            status.textContent = ev.error || "Error";
            if (typeof showToast === "function") showToast("Failed to generate.", "error");
            return;
          }
          if (ev.type === "done") {
            if (codeMessage) { codeMessage.innerHTML = ""; codeMessage.hidden = true; }
            applyDoneState(ev, lang);
            if (typeof showToast === "function") showToast("Code generated.");
            return;
          }
        }
      }
      if (buffer.startsWith("data: ")) {
        try {
          const ev = JSON.parse(buffer.slice(6));
          if (ev.type === "done") {
            if (codeMessage) { codeMessage.innerHTML = ""; codeMessage.hidden = true; }
            applyDoneState(ev, lang);
          }
        } catch (_) {}
      }
      if (typeof showToast === "function") showToast("Code generated.");
    } catch (err) {
      const errMsg = err && err.message ? err.message : "Failed to generate.";
      if (codeMessage) {
        codeMessage.classList.add("is-error");
        codeMessage.innerHTML = "<p>" + escapeHtml(errMsg) + "</p>";
        codeMessage.hidden = false;
      }
      status.textContent = errMsg;
      if (typeof showToast === "function") showToast("Failed to generate.", "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Generate Code";
    }
  });

  copyBtn?.addEventListener("click", async () => {
    const code = codeEl.textContent || "";
    if (!code.trim()) return showToast("Nothing to copy yet.");
    await navigator.clipboard.writeText(code);
    showToast("Copied code to clipboard.");
  });
});
