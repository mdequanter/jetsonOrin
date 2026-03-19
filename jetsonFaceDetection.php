<?php
session_start();

$wsBase = 'wss://signaling.ehb.be/ws/';

const APP_USERNAME = 'jetsonStayOnTrails';
const APP_PASSWORD = 'LTddk_ptxQX-omdw5B5rfpniA2wB-19KBxFaKuODMzw';

$defaultBearerToken = APP_PASSWORD;
$defaultRoom = 'jetsonDetectPersons';

if (isset($_GET['logout'])) {
    $_SESSION = [];
    session_destroy();
    header('Location: ' . strtok($_SERVER["REQUEST_URI"], '?'));
    exit;
}

$loginError = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['action']) && $_POST['action'] === 'login') {
    $username = trim($_POST['username'] ?? '');
    $password = $_POST['password'] ?? '';

    if ($username === APP_USERNAME && $password === APP_PASSWORD) {
        $_SESSION['logged_in'] = true;
    } else {
        $loginError = 'Invalid username or password.';
    }
}

$isLoggedIn = !empty($_SESSION['logged_in']);

$bearerToken = $defaultBearerToken;
$room = $defaultRoom;
$wsUrl = $wsBase . $room;
?><!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Face Assist</title>
  <style>
    :root{
      --focus:#22d3ee;
      --menu-bg:rgba(15,23,42,.9);
      --menu-border:rgba(255,255,255,.2);
      --accent:#facc15;
      --accent-ink:#111827;
      --card-bg:rgba(0,0,0,.45);
    }
    html,body{margin:0;height:100%;background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif}
    .skip-link{position:absolute;left:8px;top:-48px;z-index:5;background:var(--accent);color:var(--accent-ink);padding:8px 10px;border-radius:8px;font-weight:700;text-decoration:none}
    .skip-link:focus{top:8px}
    a:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid var(--focus);outline-offset:3px}
    .topbar{position:relative;z-index:3;background:var(--menu-bg);border-bottom:1px solid var(--menu-border)}
    .topbar-inner{max-width:1100px;margin:0 auto;padding:10px 14px;display:flex;align-items:center;justify-content:space-between;gap:12px}
    .brand{margin:0;font-size:18px;font-weight:700}
    .menu{list-style:none;margin:0;padding:0;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .menu a,.menu button{display:inline-block;color:#fff;text-decoration:none;font-weight:700;padding:7px 10px;border-radius:8px;background:transparent;border:0;cursor:pointer}
    .menu a:hover,.menu button:hover{background:rgba(255,255,255,.12)}
    .wrap{position:relative;z-index:2;min-height:calc(100% - 56px);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;padding:16px;box-sizing:border-box}
    .big{font-size:28px;font-weight:700}
    .row{opacity:.95}
    .panel{
      background:var(--card-bg);
      backdrop-filter:blur(3px);
      padding:14px 16px;
      border-radius:12px;
      border:1px solid rgba(255,255,255,.15);
      display:flex;
      flex-direction:column;
      align-items:center;
      gap:10px;
      width:min(100%, 760px);
      box-sizing:border-box;
    }
    .loginPanel{width:min(100%, 460px);align-items:stretch}
    .field{display:flex;flex-direction:column;gap:6px;width:100%}
    .field label{font-size:14px;color:#e5e7eb}
    .field input{
      padding:10px 12px;
      border-radius:10px;
      border:1px solid rgba(255,255,255,.18);
      background:#111827;
      color:#fff;
      font-size:14px;
      min-width:0;
    }
    .help{max-width:42ch;text-align:center;font-size:14px;line-height:1.45;color:#e2e8f0}
    button{
      padding:10px 14px;
      border-radius:10px;
      border:0;
      background:#222;
      color:#fff;
      font-size:14px;
      cursor:pointer;
    }
    button.primary{background:var(--accent);color:var(--accent-ink);font-weight:700}
    #video{position:fixed;inset:0;width:100vw;height:100vh;object-fit:cover;z-index:0;background:#000}
    #overlay{position:fixed;inset:0;width:100vw;height:100vh;z-index:1;pointer-events:none}
    #cap{display:none}
    .error{
      background:rgba(239,68,68,.15);
      border:1px solid rgba(239,68,68,.35);
      color:#fecaca;
      padding:10px 12px;
      border-radius:10px;
      font-size:14px;
    }
    .metaRow{
      font-size:13px;
      color:#cbd5e1;
      text-align:center;
      word-break:break-all;
    }
    form.inline{display:inline}
  </style>
</head>
<body>
<a href="#main-content" class="skip-link">Skip to main content</a>

<header class="topbar">
  <div class="topbar-inner">
    <p class="brand">AAL COMPANION - Face Assist</p>
    <nav aria-label="Main menu">
      <ul class="menu">
        <li><a href="index.php">Home</a></li>
        <?php if ($isLoggedIn): ?>
          <li>
            <form class="inline" method="get">
              <button type="submit" name="logout" value="1">Logout</button>
            </form>
          </li>
        <?php endif; ?>
      </ul>
    </nav>
  </div>
</header>

<main id="main-content">
<?php if (!$isLoggedIn): ?>
  <div class="wrap">
    <div class="panel loginPanel">
      <div class="big">Login</div>
      <p class="help">Log in to start face assist.</p>

      <?php if ($loginError): ?>
        <div class="error"><?php echo htmlspecialchars($loginError, ENT_QUOTES, 'UTF-8'); ?></div>
      <?php endif; ?>

      <form method="post" style="display:flex;flex-direction:column;gap:12px;">
        <input type="hidden" name="action" value="login" />

        <div class="field">
          <label for="username">Username</label>
          <input id="username" name="username" type="text" autocomplete="username" required />
        </div>

        <div class="field">
          <label for="password">Password</label>
          <input id="password" name="password" type="password" autocomplete="current-password" required />
        </div>

        <button class="primary" type="submit">Login</button>
      </form>
    </div>
  </div>
<?php else: ?>
<div class="wrap">
  <div class="panel">
    <div class="metaRow">
      Active WS URL: <strong><?php echo htmlspecialchars($wsUrl, ENT_QUOTES, 'UTF-8'); ?></strong>
    </div>

    <div class="big" id="status">Idle</div>
    <div class="row">Sent: <span id="sent">0</span> frames</div>
    <div class="row">Errors: <span id="errs">0</span></div>
    <div class="row">Latency: <span id="latency">--</span> ms</div>

    <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center;">
      <button id="btn">Start</button>
      <button id="switchCam" disabled>Switch Camera</button>
    </div>
  </div>
</div>

<video id="video" autoplay playsinline muted></video>
<canvas id="overlay"></canvas>
<canvas id="cap"></canvas>

<script>
(() => {
  const SIGNALING_SERVER = <?php echo json_encode($wsUrl, JSON_UNESCAPED_SLASHES); ?>;
  const BEARER_TOKEN = <?php echo json_encode($bearerToken); ?>;

  const TARGET_W = 640;
  const TARGET_H = 480;
  const JPEG_QUALITY = 0.70;
  const SEND_FPS = 2;

  const statusEl = document.getElementById("status");
  const sentEl = document.getElementById("sent");
  const errsEl = document.getElementById("errs");
  const latencyEl = document.getElementById("latency");
  const btn = document.getElementById("btn");
  const switchCamBtn = document.getElementById("switchCam");

  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const overlayCtx = overlay.getContext("2d");
  const cap = document.getElementById("cap");
  const ctx = cap.getContext("2d", { alpha: false });

  let ws = null;
  let stream = null;
  let activeVideoDeviceId = null;
  let availableVideoInputs = [];
  let timer = null;
  let sentFrames = 0;
  let errors = 0;
  let nextFrameId = 1;
  const sentAtByFrameId = new Map();
  let currentSessionId = null;
  let isAuthenticated = false;
  let authStarted = false;

  const speechCooldownMs = 5000;
  const lastSpokenAt = new Map();

  function setStatus(s) {
    statusEl.textContent = s;
  }

  function incErr() {
    errors++;
    errsEl.textContent = String(errors);
  }

  function createSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return `sess-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }

  function resizeOverlay() {
    overlay.width = window.innerWidth;
    overlay.height = window.innerHeight;
  }

  function clearOverlay() {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function mapBoxToScreen(person) {
    const videoRect = video.getBoundingClientRect();

    const scale = Math.max(
      videoRect.width / TARGET_W,
      videoRect.height / TARGET_H
    );

    const displayW = TARGET_W * scale;
    const displayH = TARGET_H * scale;

    const offsetX = videoRect.left + (videoRect.width - displayW) / 2;
    const offsetY = videoRect.top + (videoRect.height - displayH) / 2;

    return {
      x: offsetX + person.x * scale,
      y: offsetY + person.y * scale,
      w: person.w * scale,
      h: person.h * scale
    };
  }

  function drawPersons(persons) {
    clearOverlay();
    if (!Array.isArray(persons)) return;

    overlayCtx.lineWidth = 3;
    overlayCtx.font = "bold 18px Arial";
    overlayCtx.textBaseline = "top";

    for (const person of persons) {
      if (
        typeof person?.x !== "number" ||
        typeof person?.y !== "number" ||
        typeof person?.w !== "number" ||
        typeof person?.h !== "number"
      ) {
        continue;
      }

      const box = mapBoxToScreen(person);
      const name = String(person?.name || "Onbekend");
      const pos = String(person?.face_position || "").toUpperCase();

      overlayCtx.strokeStyle = "#22c55e";
      overlayCtx.fillStyle = "#22c55e";
      overlayCtx.strokeRect(box.x, box.y, box.w, box.h);

      const label = pos ? `${name} - ${pos}` : name;
      const paddingX = 8;
      const labelH = 26;
      const textW = overlayCtx.measureText(label).width;
      const labelW = textW + paddingX * 2;
      const labelY = Math.max(0, box.y - labelH);

      overlayCtx.fillRect(box.x, labelY, labelW, labelH);
      overlayCtx.fillStyle = "#000";
      overlayCtx.fillText(label, box.x + paddingX, labelY + 4);
      overlayCtx.fillStyle = "#22c55e";
    }
  }

  function sanitizeNameForFile(name) {
    return String(name || "")
      .trim()
      .replace(/\s+/g, "_")
      .replace(/[^a-zA-Z0-9_\-().]/g, "");
  }

  function positionToSuffix(facePosition) {
    const p = String(facePosition || "").toUpperCase();
    if (p === "LEFT") return "left";
    if (p === "RIGHT") return "right";
    return "front";
  }

  function getAudioPath(person) {
    const safeName = sanitizeNameForFile(person?.name || "Onbekend");
    const suffix = positionToSuffix(person?.face_position);
    return `audio/${safeName}_${suffix}.mp3`;
  }

  async function speakPerson(person) {
    const name = String(person?.name || "Onbekend");
    const pos = String(person?.face_position || "FRONT").toUpperCase();
    const key = `${name}|${pos}`;
    const now = performance.now();
    const last = lastSpokenAt.get(key) || 0;

    if ((now - last) < speechCooldownMs) return;
    lastSpokenAt.set(key, now);

    const audio = new Audio(getAudioPath(person));
    audio.preload = "auto";

    try {
      await audio.play();
    } catch (e) {
      console.log("Audio play blocked/failed:", e);
    }
  }

  async function speakPersons(persons) {
    if (!Array.isArray(persons)) return;
    for (const person of persons) {
      if (!person?.name) continue;
      await speakPerson(person);
    }
  }

  async function refreshVideoInputs() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    availableVideoInputs = devices.filter(d => d.kind === "videoinput");
    switchCamBtn.disabled = availableVideoInputs.length < 2 || !stream;
  }

  async function startCamera(preferredDeviceId = null) {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }

    const videoConstraints = preferredDeviceId
      ? {
          deviceId: { exact: preferredDeviceId },
          width: { ideal: TARGET_W },
          height: { ideal: TARGET_H }
        }
      : {
          width: { ideal: TARGET_W },
          height: { ideal: TARGET_H },
          facingMode: "environment"
        };

    stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    const track = stream.getVideoTracks()[0];
    activeVideoDeviceId = track?.getSettings?.().deviceId ?? null;
    await refreshVideoInputs();
    resizeOverlay();
  }

  function sendAuthMessage() {
    if (!ws || ws.readyState !== WebSocket.OPEN || authStarted) return;

    authStarted = true;
    setStatus("Authenticating");

    ws.send(JSON.stringify({
      type: "auth",
      token: BEARER_TOKEN
    }));
  }

  function beginAuthenticatedStreaming() {
    if (isAuthenticated) return;
    isAuthenticated = true;
    setStatus("Streaming");
    restartCaptureTimer();
  }

  function getIntervalMs() {
    return Math.round(1000 / SEND_FPS);
  }

  function restartCaptureTimer() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    if (ws && ws.readyState === WebSocket.OPEN && isAuthenticated) {
      timer = setInterval(captureAndSend, getIntervalMs());
    }
  }

  async function start() {
    btn.disabled = true;
    setStatus("Requesting camera");

    try {
      await startCamera(activeVideoDeviceId);
    } catch (e) {
      incErr();
      setStatus("Camera error");
      console.error(e);
      btn.disabled = false;
      return;
    }

    cap.width = TARGET_W;
    cap.height = TARGET_H;

    currentSessionId = createSessionId();
    isAuthenticated = false;
    authStarted = false;
    clearOverlay();

    setStatus("Connecting WS");

    try {
      ws = new WebSocket(SIGNALING_SERVER);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        btn.disabled = false;
        sendAuthMessage();
      };

      ws.onerror = (e) => {
        incErr();
        console.error("WS error", e);
      };

      ws.onclose = (e) => {
        console.warn("WS closed", e.code, e.reason);
        setStatus("WS closed");
        stop(false);
      };

      ws.onmessage = async (msg) => {
        if (typeof msg.data !== "string") return;

        try {
          const payload = JSON.parse(msg.data);

          if (payload?.type === "auth_required") {
            sendAuthMessage();
            return;
          }

          if (
            payload?.type === "auth_ok" ||
            payload?.type === "authenticated" ||
            payload?.auth === "ok" ||
            payload?.authenticated === true
          ) {
            beginAuthenticatedStreaming();
            return;
          }

          if (
            payload?.type === "auth_error" ||
            payload?.type === "unauthorized" ||
            payload?.authenticated === false
          ) {
            incErr();
            setStatus("Authentication failed");
            console.error("Authentication failed:", payload);
            stop(true);
            return;
          }

          if (payload?.type === "room_joined") {
            if (!isAuthenticated) beginAuthenticatedStreaming();
            return;
          }

          const incomingSessionId = payload?.sessionId ?? payload?.session_id ?? null;

          if (currentSessionId && incomingSessionId && incomingSessionId !== currentSessionId) {
            return;
          }

          const persons = Array.isArray(payload?.persons) ? payload.persons : [];
          drawPersons(persons);
          speakPersons(persons);

          const frameId = payload?.frame_id;
          if (frameId !== null && frameId !== undefined) {
            const id = String(frameId);
            const sentAt = sentAtByFrameId.get(id);
            if (typeof sentAt === "number") {
              const latencyMs = performance.now() - sentAt;
              latencyEl.textContent = latencyMs.toFixed(1);
              sentAtByFrameId.delete(id);
            }
          }
        } catch (e) {
          console.error("Invalid message", e);
        }
      };
    } catch (e) {
      incErr();
      setStatus("WS connect failed");
      console.error(e);
      stop(true);
    }
  }

  function stop(allowButton = true) {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }

    if (ws) {
      try { ws.close(); } catch {}
      ws = null;
    }

    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }

    switchCamBtn.disabled = true;
    sentAtByFrameId.clear();
    nextFrameId = 1;
    currentSessionId = null;
    isAuthenticated = false;
    authStarted = false;
    clearOverlay();
    latencyEl.textContent = "--";

    if (allowButton) {
      btn.disabled = false;
      btn.textContent = "Start";
      setStatus("Idle");
    }
  }

  function captureAndSend() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !isAuthenticated) return;
    if (!video.videoWidth || !video.videoHeight) return;

    ctx.drawImage(video, 0, 0, TARGET_W, TARGET_H);

    cap.toBlob(async (blob) => {
      if (!blob) return;

      try {
        const frameId = String(nextFrameId++);
        const buf = await blob.arrayBuffer();

        ws.send(JSON.stringify({
          type: "frame_meta",
          frame_id: frameId,
          sessionId: currentSessionId
        }));

        sentAtByFrameId.set(frameId, performance.now());
        ws.send(buf);

        sentFrames++;
        sentEl.textContent = String(sentFrames);

        if (sentAtByFrameId.size > 200) {
          const cutoff = performance.now() - 5000;
          for (const [id, t] of sentAtByFrameId) {
            if (t < cutoff) sentAtByFrameId.delete(id);
          }
        }
      } catch (e) {
        incErr();
        console.error(e);
      }
    }, "image/jpeg", JPEG_QUALITY);
  }

  btn.addEventListener("click", () => {
    if (timer || (ws && ws.readyState === WebSocket.OPEN)) {
      setStatus("Stopping");
      stop(true);
    } else {
      btn.textContent = "Stop";
      start();
    }
  });

  switchCamBtn.addEventListener("click", async () => {
    if (!stream) {
      setStatus("Start first");
      return;
    }

    try {
      await refreshVideoInputs();
      if (availableVideoInputs.length < 2) {
        setStatus("No extra camera found");
        return;
      }

      let idx = availableVideoInputs.findIndex(d => d.deviceId === activeVideoDeviceId);
      if (idx < 0) idx = 0;
      const next = availableVideoInputs[(idx + 1) % availableVideoInputs.length];
      await startCamera(next.deviceId);

      if (ws && ws.readyState === WebSocket.OPEN) {
        setStatus(isAuthenticated ? "Streaming" : "Authenticating");
      }
    } catch (e) {
      incErr();
      console.error("Camera switch error", e);
      setStatus("Camera switch error");
    }
  });

  window.addEventListener("resize", () => {
    resizeOverlay();
    clearOverlay();
  });

  resizeOverlay();
})();
</script>
<?php endif; ?>
</main>
</body>
</html>