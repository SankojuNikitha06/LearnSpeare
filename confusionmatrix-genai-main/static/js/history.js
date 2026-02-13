document.addEventListener("DOMContentLoaded", () => {
  const listEl = document.getElementById("historyList");
  const emptyEl = document.getElementById("historyEmpty");
  const filterEl = document.getElementById("historyFilter");
  const refreshBtn = document.getElementById("historyRefresh");
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxBackdrop = lightbox?.querySelector(".lightbox-backdrop");
  const lightboxClose = lightbox?.querySelector(".lightbox-close");
  const lightboxPrev = lightbox?.querySelector(".lightbox-prev");
  const lightboxNext = lightbox?.querySelector(".lightbox-next");
  const lightboxCounter = document.getElementById("lightboxCounter");

  let lightboxUrls = [];
  let lightboxIndex = 0;

  function showLightboxImage(index) {
    if (!lightboxImg || !lightboxUrls.length) return;
    lightboxIndex = (index + lightboxUrls.length) % lightboxUrls.length;
    lightboxImg.src = lightboxUrls[lightboxIndex];
    if (lightboxCounter) lightboxCounter.textContent = (lightboxIndex + 1) + " / " + lightboxUrls.length;
    if (lightboxPrev) lightboxPrev.style.visibility = lightboxUrls.length > 1 ? "visible" : "hidden";
    if (lightboxNext) lightboxNext.style.visibility = lightboxUrls.length > 1 ? "visible" : "hidden";
  }

  function openLightbox(urls, index) {
    if (!lightbox || !lightboxImg || !urls || !urls.length) return;
    lightboxUrls = urls.slice();
    lightboxIndex = Math.max(0, Math.min(index, urls.length - 1));
    showLightboxImage(lightboxIndex);
    lightbox.classList.add("is-open");
    lightbox.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    if (!lightbox) return;
    lightbox.classList.remove("is-open");
    lightbox.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }

  lightboxBackdrop?.addEventListener("click", closeLightbox);
  lightboxClose?.addEventListener("click", closeLightbox);
  lightboxImg?.addEventListener("click", (e) => e.stopPropagation());
  lightboxPrev?.addEventListener("click", (e) => { e.stopPropagation(); showLightboxImage(lightboxIndex - 1); });
  lightboxNext?.addEventListener("click", (e) => { e.stopPropagation(); showLightboxImage(lightboxIndex + 1); });
  document.addEventListener("keydown", (e) => {
    if (!lightbox?.classList.contains("is-open")) return;
    if (e.key === "Escape") closeLightbox();
    if (e.key === "ArrowLeft") showLightboxImage(lightboxIndex - 1);
    if (e.key === "ArrowRight") showLightboxImage(lightboxIndex + 1);
  });

  listEl?.addEventListener("click", (e) => {
    const galleryItem = e.target.closest(".history-images .gallery-item");
    if (!galleryItem) return;
    const container = galleryItem.closest(".history-images");
    if (!container) return;
    const items = container.querySelectorAll(".gallery-item img");
    const urls = Array.from(items).map((img) => img.src);
    const index = Array.from(container.querySelectorAll(".gallery-item")).indexOf(galleryItem);
    if (urls.length && index >= 0) openLightbox(urls, index);
  });

  function escapeHtml(s) {
    if (s == null) return "";
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function formatDate(iso) {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      return d.toLocaleDateString(undefined, { dateStyle: "short" }) + " " + d.toLocaleTimeString(undefined, { timeStyle: "short" });
    } catch (_) {
      return iso;
    }
  }

  function typeLabel(type) {
    const t = (type || "").toLowerCase();
    if (t === "text") return "Text";
    if (t === "code") return "Code";
    if (t === "audio") return "Audio";
    if (t === "images") return "Visuals";
    return type || "—";
  }

  function renderItem(item) {
    const payload = item.payload || {};
    const type = (item.type || "").toLowerCase();
    const id = "hist-" + (item.id || Math.random().toString(36).slice(2));
    let body = "";
    if (type === "text" && payload.content_html) {
      body = '<div class="history-content history-text">' + payload.content_html + "</div>";
    } else if (type === "code" && payload.code) {
      body = '<pre class="history-content history-code"><code>' + escapeHtml(payload.code) + "</code></pre>";
    } else if (type === "audio") {
      const script = escapeHtml((payload.script || "").slice(0, 2000));
      const url = payload.audio_url ? escapeHtml(payload.audio_url) : "";
      body = '<div class="history-content history-audio">';
      if (url) body += '<audio controls src="' + url + '"></audio>';
      body += '<pre class="history-audio-script">' + script + (payload.script && payload.script.length > 2000 ? "…" : "") + "</pre></div>";
    } else if (type === "images" && Array.isArray(payload.image_urls) && payload.image_urls.length) {
      body = '<div class="history-content history-images image-gallery">';
      payload.image_urls.forEach((u) => {
        body += '<div class="gallery-item"><img src="' + escapeHtml(u) + '" alt="diagram"/></div>';
      });
      body += "</div>";
    } else {
      body = '<p class="muted">No preview.</p>';
    }
    const recordId = item.id != null ? String(item.id) : "";
    return (
      '<div class="history-item" data-type="' + escapeHtml(type) + '" data-id="' + escapeHtml(recordId) + '">' +
      '<div class="history-head">' +
      '<button type="button" class="history-head-btn" aria-expanded="false" aria-controls="' + id + '" data-target="' + id + '">' +
      '<span class="history-type-badge history-type-' + escapeHtml(type) + '">' + escapeHtml(typeLabel(item.type)) + "</span>" +
      '<span class="history-topic">' + escapeHtml(item.topic) + "</span>" +
      '<span class="history-date">' + escapeHtml(formatDate(item.created_at)) + "</span>" +
      "</button>" +
      (recordId ? '<button type="button" class="history-delete" aria-label="Delete" data-id="' + escapeHtml(recordId) + '">&times;</button>' : "") +
      "</div>" +
      '<div id="' + id + '" class="history-body" hidden>' + body + "</div>" +
      "</div>"
    );
  }

  function loadHistory() {
    const userId = typeof getUserId === "function" ? getUserId() : "";
    if (!userId) {
      if (listEl) listEl.innerHTML = '<p class="muted">Unable to load history (no user id).</p>';
      if (emptyEl) emptyEl.style.display = "none";
      return;
    }
    const typeFilter = filterEl?.value || "";
    const url = "/api/history?user_id=" + encodeURIComponent(userId) + (typeFilter ? "&type=" + encodeURIComponent(typeFilter) : "");
    if (listEl) listEl.innerHTML = "<p class=\"muted\">Loading…</p>";
    if (emptyEl) emptyEl.style.display = "none";

    fetch(url)
      .then((r) => r.json())
      .then((data) => {
        if (!data.ok || !Array.isArray(data.items)) {
          if (listEl) listEl.innerHTML = '<p class="muted">Could not load history.</p>';
          return;
        }
        const items = data.items;
        if (items.length === 0) {
          if (listEl) listEl.innerHTML = "";
          if (emptyEl) emptyEl.style.display = "block";
          return;
        }
        if (listEl) {
          listEl.innerHTML = items.map(renderItem).join("");
          listEl.querySelectorAll(".history-head-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
              const targetId = btn.getAttribute("data-target");
              const body = targetId ? document.getElementById(targetId) : null;
              if (!body) return;
              body.hidden = !body.hidden;
              btn.setAttribute("aria-expanded", body.hidden ? "false" : "true");
            });
          });
          listEl.querySelectorAll(".history-delete").forEach((btn) => {
            btn.addEventListener("click", (e) => {
              e.preventDefault();
              e.stopPropagation();
              const recordId = btn.getAttribute("data-id");
              const itemEl = btn.closest(".history-item");
              if (!recordId || !itemEl) return;
              const userId = typeof getUserId === "function" ? getUserId() : "";
              if (!userId) return;
              fetch("/api/history/" + encodeURIComponent(recordId) + "?user_id=" + encodeURIComponent(userId), { method: "DELETE" })
                .then((r) => r.json())
                .then((data) => {
                  if (data.ok && data.deleted) itemEl.remove();
                  if (listEl && listEl.children.length === 0 && emptyEl) emptyEl.style.display = "block";
                  if (typeof showToast === "function") showToast(data.ok ? "Removed." : (data.error || "Failed"));
                })
                .catch(() => { if (typeof showToast === "function") showToast("Failed to delete.", "error"); });
            });
          });
        }
        if (emptyEl) emptyEl.style.display = "none";
      })
      .catch(() => {
        if (listEl) listEl.innerHTML = '<p class="muted">Failed to load history.</p>';
        if (emptyEl) emptyEl.style.display = "none";
      });
  }

  const clearAllBtn = document.getElementById("historyClearAll");
  const clearConfirmEl = document.getElementById("historyClearConfirm");
  const clearConfirmCancel = document.getElementById("historyClearConfirmCancel");
  const clearConfirmOk = document.getElementById("historyClearConfirmOk");

  function showClearConfirm() {
    if (clearConfirmEl) {
      clearConfirmEl.hidden = false;
      clearConfirmCancel?.focus();
    }
  }
  function hideClearConfirm() {
    if (clearConfirmEl) clearConfirmEl.hidden = true;
  }

  clearAllBtn?.addEventListener("click", showClearConfirm);
  clearConfirmCancel?.addEventListener("click", hideClearConfirm);
  clearConfirmOk?.addEventListener("click", () => {
    hideClearConfirm();
    const userId = typeof getUserId === "function" ? getUserId() : "";
    if (!userId) {
      if (typeof showToast === "function") showToast("Unable to clear.", "error");
      return;
    }
    fetch("/api/history/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId })
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.ok) {
          if (listEl) listEl.innerHTML = "";
          if (emptyEl) emptyEl.style.display = "block";
          if (typeof showToast === "function") showToast("History cleared.");
        } else if (typeof showToast === "function") showToast(data.error || "Failed", "error");
      })
      .catch(() => { if (typeof showToast === "function") showToast("Failed to clear.", "error"); });
  });

  filterEl?.addEventListener("change", loadHistory);
  refreshBtn?.addEventListener("click", loadHistory);
  loadHistory();
});
