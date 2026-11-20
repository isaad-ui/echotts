// EchoTTS frontend: no frameworks. Sends text to POST /api/tts and plays the WAV it returns.
(function () {
  "use strict";

  // Same server by default; set window.ECHOTTS_API = "https://..." to call another host.
  const API = (window.ECHOTTS_API || "").replace(/\/$/, "");
  let maxChars = 300;
  let audioUrl = null;

  const form = document.getElementById("tts-form");
  const textBox = document.getElementById("text");
  const counter = document.getElementById("counter");
  const button = document.getElementById("generate");
  const buttonLabel = button.querySelector(".label");
  const errorBox = document.getElementById("error");
  const result = document.getElementById("result");
  const player = document.getElementById("player");
  const download = document.getElementById("download");
  const status = document.getElementById("status");

  function updateCounter() {
    const n = textBox.value.length;
    counter.textContent = n + " / " + maxChars;
    counter.classList.toggle("over", n > maxChars);
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function setLoading(on) {
    button.disabled = on;
    button.classList.toggle("loading", on);
    buttonLabel.textContent = on ? "Generating…" : "Generate Speech";
  }

  async function checkHealth() {
    try {
      const r = await fetch(API + "/api/health");
      const h = await r.json();
      maxChars = h.max_chars || maxChars;
      textBox.maxLength = maxChars;
      updateCounter();
      status.textContent = h.status === "ok"
        ? "Model ready · acoustic " + h.acoustic + " · vocoder " + h.vocoder
        : "Server is up, but no model is loaded yet.";
    } catch (e) {
      status.textContent = "Cannot reach the server.";
    }
  }

  async function generate(event) {
    event.preventDefault();
    errorBox.hidden = true;
    const text = textBox.value.trim();
    if (!text) { showError("Type something to say first."); return; }
    if (text.length > maxChars) { showError("Text is too long. The limit is " + maxChars + " characters."); return; }

    setLoading(true);
    try {
      const r = await fetch(API + "/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text }),
      });
      if (!r.ok) {
        let message = "Something went wrong (" + r.status + ").";
        try { message = (await r.json()).error || message; } catch (_) { /* not JSON */ }
        showError(message);
        return;
      }
      const blob = await r.blob();
      if (audioUrl) URL.revokeObjectURL(audioUrl);   // free the previous clip
      audioUrl = URL.createObjectURL(blob);
      player.src = audioUrl;
      download.href = audioUrl;
      result.hidden = false;
      player.play().catch(function () { /* autoplay blocked: user can press play */ });
    } catch (e) {
      showError("Cannot reach the server. Check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  textBox.addEventListener("input", updateCounter);
  textBox.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) form.requestSubmit();  // Ctrl+Enter shortcut
  });
  form.addEventListener("submit", generate);
  updateCounter();
  checkHealth();
})();
