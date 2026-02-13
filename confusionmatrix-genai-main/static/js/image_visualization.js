document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("imgForm");
  const topic = document.getElementById("topic");
  const btn = document.getElementById("generateBtn");
  const promptsEl = document.getElementById("prompts");
  const gallery = document.getElementById("gallery");
  const visualsMessage = document.getElementById("visualsMessage");
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxBackdrop = lightbox?.querySelector(".lightbox-backdrop");
  const lightboxClose = lightbox?.querySelector(".lightbox-close");
  const lightboxPrev = lightbox?.querySelector(".lightbox-prev");
  const lightboxNext = lightbox?.querySelector(".lightbox-next");
  const lightboxCounter = document.getElementById("lightboxCounter");

  function escapeHtml(s) {
    if (s == null) return "";
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  let imageUrls = [];
  let currentIndex = 0;

  function showLightboxImage(index) {
    if (!lightboxImg || !imageUrls.length) return;
    currentIndex = (index + imageUrls.length) % imageUrls.length;
    lightboxImg.src = imageUrls[currentIndex];
    if (lightboxCounter) lightboxCounter.textContent = (currentIndex + 1) + " / " + imageUrls.length;
    if (lightboxPrev) lightboxPrev.style.visibility = imageUrls.length > 1 ? "visible" : "hidden";
    if (lightboxNext) lightboxNext.style.visibility = imageUrls.length > 1 ? "visible" : "hidden";
  }

  function openLightbox(urlOrIndex) {
    if (!lightbox || !lightboxImg) return;
    if (typeof urlOrIndex === "number") {
      currentIndex = urlOrIndex;
    } else {
      const idx = imageUrls.indexOf(urlOrIndex);
      currentIndex = idx >= 0 ? idx : 0;
    }
    showLightboxImage(currentIndex);
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
  lightboxPrev?.addEventListener("click", (e) => { e.stopPropagation(); showLightboxImage(currentIndex - 1); });
  lightboxNext?.addEventListener("click", (e) => { e.stopPropagation(); showLightboxImage(currentIndex + 1); });
  document.addEventListener("keydown", (e) => {
    if (!lightbox?.classList.contains("is-open")) return;
    if (e.key === "Escape") closeLightbox();
    if (e.key === "ArrowLeft") showLightboxImage(currentIndex - 1);
    if (e.key === "ArrowRight") showLightboxImage(currentIndex + 1);
  });

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const topicVal = topic.value.trim();
    if (!topicVal) return;
    btn.disabled = true;
    if (promptsEl) promptsEl.textContent = "";
    gallery.innerHTML = "";
    imageUrls = [];
    if (visualsMessage) { visualsMessage.innerHTML = ""; visualsMessage.hidden = true; }
    const badge = document.getElementById("appliedBadge");
    if (badge) badge.style.display = "none";
    const promptsList = [];
    let level = "intermediate";

    try {
      const res = await fetch("/api/generate-images-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: topicVal,
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
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let ev;
          try {
            ev = JSON.parse(line.slice(6));
          } catch (_) { continue; }
          if (ev.type === "message_only") {
            const msg = ev.message && ev.message.trim() ? ev.message.trim() : "This isn’t an ML concept I can draw. Try something like \"decision trees\", \"gradient descent\", or \"neural network layers\".";
            if (promptsEl) promptsEl.textContent = msg;
            if (ev.level) level = ev.level;
            if (badge && level) {
              const raw = level;
              badge.innerHTML = "Personalized for: " + (raw ? raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase() : "Intermediate");
              badge.style.display = "block";
            }
            if (visualsMessage) {
              visualsMessage.classList.remove("is-error");
              visualsMessage.innerHTML = "<p>" + escapeHtml(msg) + "</p>";
              visualsMessage.hidden = false;
            }
            return;
          }
          if (ev.type === "image") {
            imageUrls.push(ev.url);
            if (ev.prompt) promptsList.push(ev.prompt);
            if (promptsEl) promptsEl.textContent = promptsList.join("\n");
            const img = document.createElement("img");
            img.src = ev.url;
            img.alt = "diagram";
            const wrap = document.createElement("div");
            wrap.className = "gallery-item";
            wrap.appendChild(img);
            const imgIndex = imageUrls.length - 1;
            wrap.addEventListener("click", () => openLightbox(imgIndex));
            gallery.appendChild(wrap);
            btn.textContent = "Generating... (" + imageUrls.length + " ready)";
            if (typeof showToast === "function") showToast("Image " + imageUrls.length + " ready.");
          }
          if (ev.type === "error") {
            if (typeof showToast === "function") showToast(ev.message || "Image failed.", "error");
          }
          if (ev.type === "done") {
            if (ev.level) level = ev.level;
            if (badge && level) {
              const raw = level;
              badge.innerHTML = "Personalized for: " + (raw ? raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase() : "Intermediate");
              badge.style.display = "block";
            }
            if (imageUrls.length === 0) {
              if (visualsMessage) {
                visualsMessage.classList.remove("is-error");
                visualsMessage.innerHTML = "<p>No images were generated. Try an ML concept like \"decision trees\", \"gradient descent\", or \"neural network layers\".</p>";
                visualsMessage.hidden = false;
              }
              if (typeof showToast === "function") showToast("No images generated. Try an ML concept.", "error");
            } else {
              if (visualsMessage) { visualsMessage.innerHTML = ""; visualsMessage.hidden = true; }
              if (typeof showToast === "function") showToast("All visuals generated.");
            }
            if (imageUrls.length && typeof postJSON === "function" && typeof getUserId === "function") {
              postJSON("/api/history", {
                user_id: getUserId(),
                type: "images",
                topic: topicVal,
                payload: { image_urls: imageUrls, prompts: promptsList, level: level }
              }).catch(() => {});
            }
          }
        }
      }
    } catch (err) {
      const errMsg = err && err.message ? err.message : "Failed to generate.";
      if (promptsEl) promptsEl.textContent = errMsg;
      if (visualsMessage) {
        visualsMessage.classList.add("is-error");
        visualsMessage.innerHTML = "<p>" + escapeHtml(errMsg) + "</p>";
        visualsMessage.hidden = false;
      }
      if (typeof showToast === "function") showToast("Failed to generate.", "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Generate Visuals";
    }
  });
});
