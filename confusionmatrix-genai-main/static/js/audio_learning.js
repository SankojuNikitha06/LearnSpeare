document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("audioForm");
  const topic = document.getElementById("topic");
  const length = document.getElementById("length");
  const btn = document.getElementById("generateBtn");
  const scriptEl = document.getElementById("script");
  const player = document.getElementById("player");
  const dl = document.getElementById("downloadLink");
  const speedSelect = document.getElementById("speedSelect");

  player.playbackRate = 1;

  speedSelect?.addEventListener("change", () => {
    const speed = parseFloat(speedSelect.value);
    player.playbackRate = speed;
  });

  const audioMessage = document.getElementById("audioMessage");

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    btn.disabled = true;
    btn.textContent = "Generating...";
    scriptEl.textContent = "";
    player.src = "";
    if (audioMessage) audioMessage.textContent = "";
    const badge = document.getElementById("appliedBadge");
    if (badge) badge.style.display = "none";
    dl.href = "#";
    dl.style.display = "none";

    try {
      const data = await postJSON("/api/generate-audio", {
        topic: topic.value.trim(),
        length: length.value,
        api_key: getApiKey(),
        profile: typeof getProfile === "function" ? getProfile() : { level: "intermediate", language: "en" },
        user_id: typeof getUserId === "function" ? getUserId() : ""
      });

      if (data.message_only && data.message) {
        if (audioMessage) audioMessage.textContent = data.message;
        return;
      }

      if (audioMessage) audioMessage.textContent = "";
      scriptEl.textContent = data.script || "";
      const badge = document.getElementById("appliedBadge");
      if (badge) {
        const raw = data.level || (typeof getProfile === "function" ? getProfile().level : "intermediate");
        const level = raw ? raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase() : "Intermediate";
        badge.innerHTML = "Personalized for: " + level;
        badge.style.display = "block";
      }
      player.src = data.audio_url || "";
      if (data.download_url) {
        dl.href = data.download_url;
        dl.style.display = "inline-flex";
      }
      showToast("Audio generated.");
    } catch (err) {
      scriptEl.textContent = err.message;
      showToast("Failed to generate.", "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Generate Audio";
    }
  });
});
