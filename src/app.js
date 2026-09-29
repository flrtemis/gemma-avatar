// @ts-nocheck
/**
 * App wiring: the avatar stage + the speech-to-speech session.
 *
 * A session is one tap away: tap the button → mic → same-origin `/api/session`
 * handshake → WebSocket to the granted compute → talk. The avatar carries all
 * conversational state (listening, thinking, speaking) with its body; the
 * caption under it is a quiet machine-voice echo of the same state.
 */

/** @typedef {any} HTMLElement */
/** @typedef {any} HTMLButtonElement */
/** @typedef {any} HTMLInputElement */
/** @typedef {any} HTMLDialogElement */
/** @typedef {any} HTMLSelectElement */
/** @typedef {any} HTMLTextAreaElement */
/** @typedef {any} AudioContext */

import { S2sWsRealtimeClient } from "./s2s/s2s-ws-client.js";
import { AvatarStage, AVATAR_MOODS, AVATAR_GESTURES } from "./avatar.js";

const VOICES = [
  "Aiden",
  "Ryan",
  "Dylan",
  "Eric",
  "Ono_Anna",
  "Serena",
  "Sohee",
  "Uncle_Fu",
  "Vivian",
];
const DEFAULT_VOICE = "Ono_Anna";

const DEFAULT_INSTRUCTIONS = [
  "You are a friendly voice assistant with a visible, human-like 3D avatar: the user",
  "sees you as a person on their screen. This is a spoken conversation: keep replies",
  "short, natural and warm, never list-like.",
  "You can control your avatar body with tools: set_mood changes your overall emotional",
  "state, make_hand_gesture plays a hand gesture, make_facial_expression makes a quick",
  "facial expression from a single face emoji. Use them naturally and sparingly to",
  "express yourself: smile when greeting, shrug when unsure, thumbs up when agreeing.",
  "Never mention the tools or that you are controlling an avatar.",
].join(" ");

const STORAGE_KEYS = {
  voice: "avatar.voice",
  instructions: "avatar.instructions",
  directUrl: "avatar.directUrl",
  subtitles: "avatar.subtitles",
};

/** Function tools declared to the backend: the model plays the avatar. */
/** @type {import("./s2s/s2s-ws-client.js").ToolDef[]} */
const TOOL_DEFS = [
  {
    type: /** @type {"function"} */ ("function"),
    name: "set_mood",
    description: "Change your avatar's overall mood/emotional state.",
    parameters: {
      type: "object",
      properties: {
        mood: { type: "string", enum: AVATAR_MOODS, description: "Mood name." },
      },
      required: ["mood"],
    },
  },
  {
    type: /** @type {"function"} */ ("function"),
    name: "make_hand_gesture",
    description: "Make a hand gesture with your avatar.",
    parameters: {
      type: "object",
      properties: {
        gesture: { type: "string", enum: AVATAR_GESTURES, description: "Gesture name." },
      },
      required: ["gesture"],
    },
  },
  {
    type: /** @type {"function"} */ ("function"),
    name: "make_facial_expression",
    description: "Make a quick facial expression with your avatar, given as a single face emoji (e.g. 😊, 😮, 🤔).",
    parameters: {
      type: "object",
      properties: {
        emoji: { type: "string", description: "A single face emoji." },
      },
      required: ["emoji"],
    },
  },
];

// ── DOM ──────────────────────────────────────────────────────────────────
/** @param {string} sel */
const $ = (sel) => /** @type {HTMLElement} */ (document.querySelector(sel));
const stageNode = $("#stage");
const mainBtn = /** @type {HTMLButtonElement} */ ($("#main-btn"));
const mainBtnLabel = $("#main-btn-label");
const attachBtn = /** @type {HTMLButtonElement} */ ($("#attach-btn"));
const muteBtn = /** @type {HTMLButtonElement} */ ($("#mute-btn"));
const uploadInput = /** @type {HTMLInputElement} */ ($("#upload-input"));
const promptForm = /** @type {HTMLFormElement} */ ($("#prompt-form"));
const promptInput = /** @type {HTMLTextAreaElement} */ ($("#prompt-input"));
const promptSend = /** @type {HTMLButtonElement} */ ($("#prompt-send"));
const caption = $("#caption");
const subtitles = $("#subtitles");
const loading = $("#loading");
const settingsBtn = /** @type {HTMLButtonElement} */ ($("#settings-btn"));
const settingsDialog = /** @type {HTMLDialogElement} */ ($("#settings"));
const inputVoice = /** @type {HTMLSelectElement} */ ($("#voice"));
const inputInstructions = /** @type {HTMLTextAreaElement} */ ($("#instructions"));
const inputDirectUrl = /** @type {HTMLInputElement} */ ($("#direct-url"));
const inputSubtitles = /** @type {HTMLInputElement} */ ($("#subtitles-toggle"));
const directUrlRow = $("#direct-url-row");

// ── State ────────────────────────────────────────────────────────────────
const stage = new AvatarStage(stageNode);
/** @type {S2sWsRealtimeClient | null} */
let client = null;
let muted = false;
let subtitleTimer = 0;
/** @type {{ lb: boolean, allowDirect: boolean }} */
let config = { lb: false, allowDirect: true };

function loadSettings() {
  return {
    voice: localStorage.getItem(STORAGE_KEYS.voice) || DEFAULT_VOICE,
    instructions: localStorage.getItem(STORAGE_KEYS.instructions) || "",
    directUrl:
      localStorage.getItem(STORAGE_KEYS.directUrl) ||
      "ws://localhost:8765/v1/realtime",
    subtitles: localStorage.getItem(STORAGE_KEYS.subtitles) !== "0",
  };
}
let settings = loadSettings();

function saveSettings() {
  localStorage.setItem(STORAGE_KEYS.voice, settings.voice);
  localStorage.setItem(STORAGE_KEYS.instructions, settings.instructions);
  localStorage.setItem(STORAGE_KEYS.directUrl, settings.directUrl);
  localStorage.setItem(STORAGE_KEYS.subtitles, settings.subtitles ? "1" : "0");
}

/** Persona + whatever extra guidance the user typed in Settings. */
function effectiveInstructions() {
  const extra = settings.instructions.trim();
  return extra ? `${DEFAULT_INSTRUCTIONS}\n\nAdditional instructions from the user:\n${extra}` : DEFAULT_INSTRUCTIONS;
}

// ── Captions / subtitles ─────────────────────────────────────────────────
/** @param {string} text @param {""|"live"|"error"} [kind] */
function setCaption(text, kind = "") {
  caption.textContent = text;
  caption.className = kind;
}

/** @param {string} text */
function showSubtitles(text) {
  if (!settings.subtitles) return;
  clearTimeout(subtitleTimer);
  subtitles.textContent = text;
  subtitles.classList.add("visible");
}

function fadeSubtitles(delayMs = 2600) {
  clearTimeout(subtitleTimer);
  subtitleTimer = window.setTimeout(() => subtitles.classList.remove("visible"), delayMs);
}

/** @param {File} file */
function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const out = String(reader.result || "");
      const base64 = out.includes(",") ? out.slice(out.indexOf(",") + 1) : out;
      resolve(base64);
    };
    reader.onerror = () => reject(new Error("file read failed"));
    reader.readAsDataURL(file);
  });
}


/** @param {File} file */
async function analyzeUpload(file) {
  if (!client || client.status === "closed" || client.status === "error") {
    setCaption("START TALKING FIRST, THEN ATTACH A FILE", "error");
    return;
  }
  const mime = file.type || "application/octet-stream";
  if (!mime.startsWith("image/") && !mime.startsWith("video/")) {
    setCaption("ONLY IMAGE OR VIDEO FILES ARE SUPPORTED", "error");
    return;
  }

  attachBtn.disabled = true;

  try {
    const base64 = await readFileAsBase64(file);
    client.sendUserFile(base64, file.name, mime);
    client.requestResponse();
  } catch (err) {
    console.error("upload analysis failed", err);
    setCaption("UPLOAD FAILED, TRY ANOTHER FILE", "error");
  } finally {
    attachBtn.disabled = false;
  }
}

// ── Button ───────────────────────────────────────────────────────────────
/** @type {"start" | "join" | "stop" | "busy"} */
let mainAction = "start";

/** @param {"start" | "join" | "stop" | "busy"} action @param {string} label */
function setMainButton(action, label) {
  mainAction = action;
  mainBtnLabel.textContent = label;
  mainBtn.disabled = action === "busy";
  mainBtn.classList.toggle("live", action === "stop");
  muteBtn.hidden = action !== "stop";
}

// ── Status handling ──────────────────────────────────────────────────────
/** @type {Record<string, string>} */
const CAPTIONS = {
  idle: "TAP TO TALK",
  "creating-session": "REQUESTING A SLOT…",
  queued: "WAITING IN LINE…",
  "your-turn": "YOUR TURN, TAP TO JOIN",
  connecting: "CONNECTING…",
  connected: "GO AHEAD, I'M LISTENING",
  "user-speaking": "LISTENING",
  processing: "THINKING…",
  "ai-speaking": "SPEAKING",
  closed: "TAP TO TALK",
  error: "SOMETHING BROKE, TAP TO RETRY",
};

/** @param {string} status */
function onStatus(status) {
  stage.setConversationState(status);
  setCaption(CAPTIONS[status] ?? status, status === "error" ? "error" : status === "idle" || status === "closed" ? "" : "live");

  switch (status) {
    case "idle":
    case "closed":
      setMainButton("start", "Start talking");
      break;
    case "error":
      setMainButton("start", "Retry");
      break;
    case "creating-session":
    case "connecting":
      setMainButton("busy", "Connecting…");
      break;
    case "queued":
      setMainButton("stop", "Leave queue");
      break;
    case "your-turn":
      setMainButton("join", "Join now");
      break;
    default:
      // connected / user-speaking / processing / ai-speaking
      setMainButton("stop", "End conversation");
      break;
  }

  if (status === "user-speaking") {
    subtitles.classList.remove("visible");
  }
}

// ── Tool executor ────────────────────────────────────────────────────────
/** @param {string} name @param {string} argsJson @param {string} callId */
function runTool(name, argsJson, callId) {
  if (!client) return;
  /** @type {Record<string, unknown>} */
  let args = {};
  try {
    args = JSON.parse(argsJson || "{}");
  } catch {
    // keep {}
  }
  const result = stage.runTool(name, args) ?? `Unknown tool: ${name}`;
  client.sendToolOutput(callId, result);
  // The turn continues after a tool call only when we ask for the follow-up.
  client.requestResponse();
}

// ── Session lifecycle ────────────────────────────────────────────────────
async function startSession() {
  // Everything audible hangs off the avatar's AudioContext; resume it inside
  // the tap gesture or iOS keeps it suspended (silent).
  stage.resume();

  let micStream;
  if (new URLSearchParams(location.search).has("fakemic")) {
    // Dev/testing hook: a silent synthetic mic, so the session can be driven
    // end-to-end (handshake, WS, TTS playback, lip-sync) without a real mic
    // or a native permission prompt.
    const ctx = /** @type {AudioContext} */ (stage.audioCtx);
    micStream = ctx.createMediaStreamDestination().stream;
  } else {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch {
      setCaption("MIC BLOCKED, ALLOW IT IN THE BROWSER AND RETRY", "error");
      return;
    }
  }

  const audioCtx = stage.audioCtx;
  const voiceSink = stage.voiceSink;
  if (!audioCtx || !voiceSink) return;

  const c = new S2sWsRealtimeClient({
    ...(config.lb ? { sessionUrl: "api/session" } : { directUrl: settings.directUrl }),
    voice: settings.voice,
    instructions: effectiveInstructions(),
    micStream,
    audioContext: audioCtx,
    outputNode: voiceSink,
    workletBaseUrl: "/worklets/",
    tools: TOOL_DEFS,
  });
  client = c;

  c.addEventListener("status", (e) => onStatus(/** @type {CustomEvent} */ (e).detail.status));

  c.addEventListener("queue", (e) => {
    const { position } = /** @type {CustomEvent} */ (e).detail;
    setCaption(position > 0 ? `#${position} IN LINE…` : "ALMOST THERE…", "live");
  });

  c.addEventListener("transcript", (e) => {
    const { role, text } = /** @type {CustomEvent} */ (e).detail;
    if (role === "assistant" && text) showSubtitles(text);
  });

  c.addEventListener("response-finished", () => {
    fadeSubtitles();
  });

  c.addEventListener("toolcall", (e) => {
    const { name, arguments: args, callId } = /** @type {CustomEvent} */ (e).detail;
    runTool(name, args, callId);
  });

  c.addEventListener("server-error", (e) => {
    const error = /** @type {CustomEvent} */ (e).detail.error;
    console.warn("server error:", error);
  });

  c.addEventListener("error", () => {
    void endSession();
  });

  try {
    await c.connect();
  } catch (err) {
    const code = /** @type {Error & {code?: string}} */ (err)?.code;
    if (code === "limit") {
      setCaption("DAILY CONVERSATION LIMIT REACHED, TRY AGAIN TOMORROW", "error");
    } else if (code === "queue-full") {
      setCaption("EVERY SEAT IS TAKEN, TRY AGAIN SHORTLY", "error");
    } else if (code === "join-expired") {
      setCaption("YOUR SPOT EXPIRED, TAP TO TRY AGAIN", "error");
    } else if (code !== "aborted") {
      console.error(err);
      setCaption("COULD NOT CONNECT, TAP TO RETRY", "error");
    }
    await endSession(true);
    return;
  }
}

/** @param {boolean} [silent] Keep the current caption (e.g. an error). */
async function endSession(silent = false) {
  const c = client;
  client = null;
  if (c) {
    for (const track of c.options.micStream?.getTracks() ?? []) track.stop();
    await c.close().catch(() => {});
  }
  stage.setConversationState("idle");
  subtitles.classList.remove("visible");
  if (!silent) setCaption(CAPTIONS.idle);
  setMainButton("start", "Start talking");
}

// ── Text messaging ───────────────────────────────────────────────────────
async function sendTypedPrompt() {
  const text = promptInput.value.trim();
  if (!text) return;

  if (!client || client.status === "closed" || client.status === "error") {
    await startSession();
    if (!client || client.status === "closed" || client.status === "error") {
      return;
    }
  }

  promptInput.value = "";
  promptInput.style.height = "auto";
  client.sendUserText(text);
  client.requestResponse();
}

// ── UI events ────────────────────────────────────────────────────────────
promptForm?.addEventListener("submit", (e) => {
  e.preventDefault();
  void sendTypedPrompt();
});

promptInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    void sendTypedPrompt();
  }
});

promptInput?.addEventListener("input", () => {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 120)}px`;
});

mainBtn.addEventListener("click", () => {
  if (mainAction === "start") void startSession();
  else if (mainAction === "join") {
    stage.resume(); // fresh gesture: re-arm audio before dialing
    client?.join();
  } else if (mainAction === "stop") void endSession();
});

muteBtn.addEventListener("click", () => {
  muted = !muted;
  client?.setMuted(muted);
  muteBtn.classList.toggle("active", muted);
  muteBtn.setAttribute("aria-label", muted ? "Unmute microphone" : "Mute microphone");
});

attachBtn.addEventListener("click", () => {
  uploadInput.click();
});

uploadInput.addEventListener("change", () => {
  const file = uploadInput.files?.[0];
  if (file) void analyzeUpload(file);
  uploadInput.value = "";
});

settingsBtn.addEventListener("click", () => {
  inputVoice.value = settings.voice;
  inputInstructions.value = settings.instructions;
  inputDirectUrl.value = settings.directUrl;
  inputSubtitles.checked = settings.subtitles;
  settingsDialog.showModal();
});

settingsDialog.addEventListener("close", () => {
  settings = {
    voice: inputVoice.value || DEFAULT_VOICE,
    instructions: inputInstructions.value,
    directUrl: inputDirectUrl.value.trim(),
    subtitles: inputSubtitles.checked,
  };
  saveSettings();
  if (!settings.subtitles) subtitles.classList.remove("visible");
  // Voice/instructions apply live to an ongoing session.
  client?.updateSession({ voice: settings.voice, instructions: effectiveInstructions() });
});

window.addEventListener("beforeunload", () => {
  client?.close();
});

// ── Boot ─────────────────────────────────────────────────────────────────
async function boot() {
  for (const v of VOICES) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v.replaceAll("_", " ");
    inputVoice.append(o);
  }

  try {
    const resp = await fetch("api/config");
    if (resp.ok) config = { ...config, ...(await resp.json()) };
  } catch {
    // defaults keep direct mode available
  }
  directUrlRow.hidden = !config.allowDirect;

  setCaption("WAKING HER UP…");
  setMainButton("busy", "Loading…");
  try {
    await stage.init({
      onprogress: (ev) => {
        if (ev.lengthComputable) {
          const pct = Math.min(100, Math.round((ev.loaded / ev.total) * 100));
          loading.textContent = `Loading avatar ${pct}%`;
        }
      },
    });
  } catch (err) {
    console.error(err);
    loading.textContent = "The avatar failed to load. Check the console and reload.";
    setCaption("AVATAR FAILED TO LOAD", "error");
    return;
  }
  loading.classList.add("done");
  setCaption(CAPTIONS.idle);
  setMainButton("start", "Start talking");
  attachBtn.disabled = false;

  // Debug handles
  Object.assign(window, { stage, getClient: () => client });
}

void boot();
