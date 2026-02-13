document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("textForm");
  const topic = document.getElementById("topic");
  const depth = document.getElementById("depth");
  const btn = document.getElementById("generateBtn");
  const out = document.getElementById("output");
  const copyBtn = document.getElementById("copyBtn");
  const textMessage = document.getElementById("textMessage");

  function applyDoneState(data) {
    out.innerHTML = data.content_html || "";
    out.dataset.md = data.content_md || "";
    out.classList.remove("streaming-output");
    const badge = document.getElementById("appliedBadge");
    if (badge) {
      const raw = data.level || (typeof getProfile === "function" ? getProfile().level : "intermediate");
      const level = raw ? raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase() : "Intermediate";
      let html = "Personalized for: " + level;
      if (data.rag_used) html += " &middot; Grounded: Yes (scikit-learn, experimental)";
      badge.innerHTML = html;
      badge.style.display = "block";
    }
    const sourcesWrap = document.getElementById("sourcesWrap");
    const sourcesEl = document.getElementById("sources");
    if (sourcesWrap && sourcesEl) {
      if (data.rag_used && Array.isArray(data.citations) && data.citations.length > 0) {
        sourcesWrap.style.display = "block";
        sourcesEl.innerHTML = data.citations.map((c, i) => {
          const preview = (c.text || "").slice(0, 200).replace(/\n/g, " ");
          const score = typeof c.score === "number" ? c.score.toFixed(3) : "";
          return `<div class="source-item">
            <strong>[${c.rank || i + 1}]</strong> ${escapeHtml(c.source || "")} ${c.title ? " — " + escapeHtml(c.title) : ""} ${score ? " <span class=\"source-score\">(score " + score + ")</span>" : ""}
            <div class="source-snippet">${escapeHtml(preview)}${(c.text || "").length > 200 ? "…" : ""}</div>
          </div>`;
        }).join("");
      } else {
        sourcesWrap.style.display = "none";
        sourcesEl.innerHTML = "";
      }
    }
  }

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const topicVal = topic?.value?.trim() || "";
    if (!topicVal) {
      if (textMessage) {
        textMessage.innerHTML = "<p>Please enter a topic (e.g. backpropagation in neural networks).</p>";
        textMessage.classList.remove("is-error");
        textMessage.hidden = false;
      }
      topic?.focus();
      if (typeof showToast === "function") showToast("Enter a topic first.");
      return;
    }
    out.innerHTML = "";
    if (textMessage) { textMessage.innerHTML = ""; textMessage.hidden = true; }
    const badge = document.getElementById("appliedBadge");
    if (badge) badge.style.display = "none";
    const sourcesWrap = document.getElementById("sourcesWrap");
    if (sourcesWrap) sourcesWrap.style.display = "none";
    btn.disabled = true;
    btn.textContent = "Generating...";
    out.classList.add("streaming-output");

    try {
      const res = await fetch("/api/generate-text-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: topicVal,
          depth: depth?.value || "standard",
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
            const msg = (ev.message && ev.message.trim()) || "LearnSphere is for ML learning—try \"backpropagation\", \"decision trees\", or \"bias-variance tradeoff\".";
            if (textMessage) {
              textMessage.classList.remove("is-error");
              textMessage.innerHTML = "<p>" + escapeHtml(msg) + "</p>";
              textMessage.hidden = false;
            }
            return;
          }
          if (ev.type === "chunk" && ev.text) {
            accumulated += ev.text;
            out.innerHTML = escapeHtml(accumulated);
            out.classList.add("streaming-output");
          }
          if (ev.type === "error") {
            if (textMessage) {
              textMessage.classList.add("is-error");
              textMessage.innerHTML = "<p>" + escapeHtml(ev.error || "Error") + "</p>";
              textMessage.hidden = false;
            }
            out.classList.remove("streaming-output");
            if (typeof showToast === "function") showToast("Failed to generate.", "error");
            return;
          }
          if (ev.type === "done") {
            if (textMessage) { textMessage.innerHTML = ""; textMessage.hidden = true; }
            applyDoneState(ev);
            if (typeof showToast === "function") showToast("Explanation generated.");
            return;
          }
        }
      }
      if (buffer.startsWith("data: ")) {
        try {
          const ev = JSON.parse(buffer.slice(6));
          if (ev.type === "done") {
            if (textMessage) { textMessage.innerHTML = ""; textMessage.hidden = true; }
            applyDoneState(ev);
          }
        } catch (_) {}
      }
      if (typeof showToast === "function") showToast("Explanation generated.");
    } catch (err) {
      const errMsg = err && err.message ? err.message : "Failed to generate.";
      if (textMessage) {
        textMessage.classList.add("is-error");
        textMessage.innerHTML = "<p>" + escapeHtml(errMsg) + "</p>";
        textMessage.hidden = false;
      }
      out.classList.remove("streaming-output");
      if (typeof showToast === "function") showToast("Failed to generate.", "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Generate Explanation";
    }
  });

  copyBtn?.addEventListener("click", async () => {
    const md = out.dataset.md || "";
    if (!md) return showToast("Nothing to copy yet.");
    await navigator.clipboard.writeText(md);
    showToast("Copied markdown to clipboard.");
  });
});
