/**
 * Onboarding: show welcome modal for first-time visitors to set level and personalization.
 */
document.addEventListener("DOMContentLoaded", () => {
  const modal = document.getElementById("onboardingModal");
  const form = document.getElementById("onboardingForm");
  const levelSelect = document.getElementById("onboardingLevel");
  const focusSelect = document.getElementById("onboardingFocus");
  const ragCheck = document.getElementById("onboardingRag");
  const backdrop = modal?.querySelector(".onboarding-backdrop");

  function showModal() {
    if (!modal) return;
    modal.classList.add("is-open");
    modal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  }

  function hideModal() {
    if (!modal) return;
    modal.classList.remove("is-open");
    modal.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }

  if (typeof getOnboardingDone === "function" && getOnboardingDone()) {
    return;
  }
  showModal();

  backdrop?.addEventListener("click", hideModal);

  document.getElementById("onboardingSkip")?.addEventListener("click", () => {
    if (typeof setOnboardingDone === "function") setOnboardingDone();
    hideModal();
  });

  form?.addEventListener("submit", (e) => {
    e.preventDefault();
    if (typeof setLevel === "function" && levelSelect) {
      setLevel(levelSelect.value || "intermediate");
    }
    if (typeof setRagEnabled === "function" && ragCheck) {
      setRagEnabled(ragCheck.checked);
    }
    if (typeof setFocus === "function" && focusSelect) {
      setFocus(focusSelect.value || "both");
    }
    if (typeof setOnboardingDone === "function") {
      setOnboardingDone();
    }
    hideModal();
    if (typeof showToast === "function") {
      showToast("Profile saved. You can change this anytime in Settings.");
    }
  });
});
