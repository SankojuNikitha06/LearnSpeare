document.addEventListener("DOMContentLoaded", async () => {
  const input = document.getElementById("apiKey");
  const saveBtn = document.getElementById("saveKey");
  const clearBtn = document.getElementById("clearKey");
  const status = document.getElementById("keyStatus");
  const modelInfo = document.getElementById("modelInfo");
  const learningLevel = document.getElementById("learningLevel");
  const saveProfileBtn = document.getElementById("saveProfile");
  const ragEnabled = document.getElementById("ragEnabled");

  if (input) input.value = getApiKey();

  if (typeof getProfile === "function") {
    const profile = getProfile();
    if (learningLevel) learningLevel.value = profile.level || "intermediate";
    if (ragEnabled) ragEnabled.checked = profile.rag_enabled === true;
  }
  function saveProfile() {
    if (typeof setLevel !== "function" || typeof setRagEnabled !== "function") return;
    const level = learningLevel?.value || "intermediate";
    setLevel(level);
    setRagEnabled(ragEnabled?.checked !== false);
    showToast("Saved locally.");
  }
  saveProfileBtn?.addEventListener("click", saveProfile);
  learningLevel?.addEventListener("change", saveProfile);
  ragEnabled?.addEventListener("change", saveProfile);

  const refreshStatus = () => {
    const k = getApiKey();
    status.textContent = k ? "Saved in this browser (localStorage)." : "Not set.";
  };

  refreshStatus();

  saveBtn?.addEventListener("click", () => {
    const key = input.value.trim();
    if (typeof setApiKey !== "function") {
      status.textContent = "Error: main.js not loaded.";
      return;
    }
    const ok = setApiKey(key);
    refreshStatus();
    if (ok) {
      if (input) input.value = getApiKey();
      showToast("API key saved.");
    } else {
      status.textContent = "Save failed. Try disabling private browsing or allow localStorage for this site.";
      if (typeof showToast === "function") showToast("Could not save key.");
    }
  });

  clearBtn?.addEventListener("click", () => {
    setApiKey("");
    input.value = "";
    refreshStatus();
    showToast("API key cleared.");
  });

  // Model info (server-side)
  try {
    const resp = await fetch("/api/model-info");
    const data = await resp.json();
    if (data.ok) {
      modelInfo.textContent = JSON.stringify(data.models, null, 2);
    } else {
      modelInfo.textContent = data.error || "Unable to load model info.";
    }
  } catch (e) {
    modelInfo.textContent = String(e);
  }
});
