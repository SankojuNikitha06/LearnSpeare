/**
 * Local "DB" for user profile using localStorage key learnsphere_profile.
 */

const PROFILE_KEY = "learnsphere_profile";
const ONBOARDING_KEY = "learnsphere_onboarding_done";
const USER_ID_KEY = "learnsphere_user_id";
const LEVELS = ["beginner", "intermediate", "advanced"];

function getUserId() {
  try {
    let id = localStorage.getItem(USER_ID_KEY);
    if (id && id.length > 10) return id;
    id = typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : "ls-" + Date.now() + "-" + Math.random().toString(36).slice(2, 12);
    localStorage.setItem(USER_ID_KEY, id);
    return id;
  } catch (_) {
    return "ls-anon-" + Date.now();
  }
}

const FOCUS_OPTIONS = ["concepts", "code", "both"];

function getProfile() {
  try {
    const raw = localStorage.getItem(PROFILE_KEY);
    if (!raw) return { level: "intermediate", language: "en", rag_enabled: false, focus: "both" };
    const p = JSON.parse(raw);
    const level = LEVELS.includes(p.level) ? p.level : "intermediate";
    const rag_enabled = typeof p.rag_enabled === "boolean" ? p.rag_enabled : false;
    const focus = FOCUS_OPTIONS.includes(p.focus) ? p.focus : "both";
    return { level, language: "en", rag_enabled, focus };
  } catch (_) {
    return { level: "intermediate", language: "en", rag_enabled: false, focus: "both" };
  }
}

function getLevel() {
  return getProfile().level;
}

function setLevel(level) {
  const normalized = (level || "").toLowerCase().trim();
  if (!LEVELS.includes(normalized)) return false;
  const profile = getProfile();
  profile.level = normalized;
  localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
  return true;
}

function getRagEnabled() {
  return getProfile().rag_enabled;
}

function setRagEnabled(enabled) {
  const profile = getProfile();
  profile.rag_enabled = !!enabled;
  localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
  return true;
}

function setFocus(focus) {
  const normalized = (focus || "both").toLowerCase().trim();
  if (!FOCUS_OPTIONS.includes(normalized)) return false;
  const profile = getProfile();
  profile.focus = normalized;
  localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
  return true;
}

function getOnboardingDone() {
  return localStorage.getItem(ONBOARDING_KEY) === "true";
}

function setOnboardingDone() {
  localStorage.setItem(ONBOARDING_KEY, "true");
}

function resetProfile() {
  localStorage.removeItem(PROFILE_KEY);
  localStorage.removeItem(ONBOARDING_KEY);
}
