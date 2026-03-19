<?php
session_start();

/*
  Simple demo login.
  Change these before deploying.
*/

/*
  Default values shown after login.
*/
$wsBase = 'wss://signaling.ehb.be/ws/';


const APP_USERNAME = 'jetsonStayOnTrails';
const APP_PASSWORD = 'LTddk_ptxQX-omdw5B5rfpniA2wB-19KBxFaKuODMzw';

$defaultBearerToken = APP_PASSWORD;
$defaultRoom =  APP_USERNAME;


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

if ($isLoggedIn) {
    if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['action']) && $_POST['action'] === 'settings') {
        $bearerToken = trim($_POST['bearer_token'] ?? $defaultBearerToken);
        $room = trim($_POST['ws_room'] ?? $defaultRoom);

        if ($room === '') {
            $room = $defaultRoom;
        }

        $room = preg_replace('~^/ws/~', '', $room);
        $room = trim($room, '/');

        $_SESSION['bearer_token'] = $bearerToken;
        $_SESSION['ws_room'] = $room;
    } else {
        if (isset($_SESSION['bearer_token'])) {
            $bearerToken = $_SESSION['bearer_token'];
        }
        if (isset($_SESSION['ws_room'])) {
            $room = $_SESSION['ws_room'];
        }
    }
}

$wsUrl = $wsBase . $room;
?><!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stay On Trails - Jetson</title>
  <style>
    :root{
      --focus:#22d3ee;
      --menu-bg:rgba(15,23,42,.9);
      --menu-border:rgba(255,255,255,.2);
      --accent:#facc15;
      --accent-ink:#111827;
      --card-bg:rgba(0,0,0,.45);
      --input-bg:#111827;
      --input-border:rgba(255,255,255,.18);
    }
    html,body{margin:0;height:100%;background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif}
    .skip-link{position:absolute;left:8px;top:-48px;z-index:5;background:var(--accent);color:var(--accent-ink);padding:8px 10px;border-radius:8px;font-weight:700;text-decoration:none}
    .skip-link:focus{top:8px}
    a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid var(--focus);outline-offset:3px}
    .topbar{position:relative;z-index:3;background:var(--menu-bg);border-bottom:1px solid var(--menu-border)}
    .topbar-inner{max-width:1100px;margin:0 auto;padding:10px 14px;display:flex;align-items:center;justify-content:space-between;gap:12px}
    .brand{margin:0;font-size:18px;font-weight:700}
    .menu{list-style:none;margin:0;padding:0;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .menu a,.menu button{display:inline-block;color:#fff;text-decoration:none;font-weight:700;padding:7px 10px;border-radius:8px;background:transparent;border:0;cursor:pointer}
    .menu a:hover,.menu button:hover{background:rgba(255,255,255,.12)}
    .menu .cta{background:var(--accent);color:var(--accent-ink)}
    .menu .cta:hover{background:#fde047}
    .wrap{position:relative;z-index:2;min-height:calc(100% - 56px);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;padding:16px;box-sizing:border-box}
    .big{font-size:28px;font-weight:700}
    .row{opacity:.9}
    .compassWrap{display:flex;flex-direction:column;align-items:center;gap:6px}
    #compass{width:140px;height:140px;display:block}
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
      width:min(100%, 880px);
      box-sizing:border-box;
    }
    .loginPanel{
      width:min(100%, 460px);
      align-items:stretch;
    }
    .settingsGrid{
      display:grid;
      grid-template-columns:1fr 1fr auto;
      gap:10px;
      width:100%;
      align-items:end;
    }
    .field{display:flex;flex-direction:column;gap:6px}
    .field label{font-size:14px;color:#e5e7eb}
    .field input{
      padding:10px 12px;
      border-radius:10px;
      border:1px solid var(--input-border);
      background:var(--input-bg);
      color:#fff;
      font-size:14px;
      min-width:0;
    }
    .help{max-width:42ch;text-align:center;font-size:14px;line-height:1.45;color:#e2e8f0}
    .help a{color:#93c5fd}
    .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    button,select{
      padding:10px 14px;
      border-radius:10px;
      border:0;
      background:#222;
      color:#fff;
      font-size:14px;
    }
    button.primary{background:var(--accent);color:var(--accent-ink);font-weight:700}
    .fpsControl{display:flex;align-items:center;gap:8px}
    .modelControl{display:flex;flex-direction:column;gap:6px;align-items:flex-start}
    .modelOptions{display:flex;gap:12px;flex-wrap:wrap}
    .modelOptions label{display:flex;align-items:center;gap:6px;font-size:14px}
    .modelOptions input{margin:0}
    #video{position:fixed;inset:0;width:100vw;height:100vh;object-fit:cover;z-index:0;background:#000}
    #cap{display:none;}
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
    @media (max-width: 760px){
      .settingsGrid{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
<a href="#main-content" class="skip-link">Skip to main content</a>

<header class="topbar">
  <div class="topbar-inner">
    <p class="brand">AAL COMPANION - Stay on trails</p>
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
      <p class="help">If you have an AAL Companion box and you have a valid username and password, please log in below.</p>

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
    <form method="post" class="settingsGrid">
      <input type="hidden" name="action" value="settings" />

      <div class="field">
        <label for="bearer_token">Bearer token</label>
        <input
          id="bearer_token"
          name="bearer_token"
          type="text"
          value="<?php echo htmlspecialchars($bearerToken, ENT_QUOTES, 'UTF-8'); ?>"
          placeholder="Enter bearer token"
        />
      </div>

      <div class="field">
        <label for="ws_room">WebSocket room</label>
        <input
          id="ws_room"
          name="ws_room"
          type="text"
          value="<?php echo htmlspecialchars($room, ENT_QUOTES, 'UTF-8'); ?>"
          placeholder="Example: jetsonStayOnTrails"
        />
      </div>

      <button class="primary" type="submit">Save</button>
    </form>

    <div class="metaRow">
      Active WS URL: <strong><?php echo htmlspecialchars($wsUrl, ENT_QUOTES, 'UTF-8'); ?></strong>
    </div>

    <div class="big" id="status">Idle</div>
    <div class="row">Sent: <span id="sent">0</span> frames <span id="kbps">0</span> kbps</div>
    <div class="row">Errors: <span id="errs">0</span></div>
    <div class="row">Latency: <span id="latency">--</span> ms</div>

    <fieldset class="modelControl">
      <legend>Model</legend>
      <div class="modelOptions">
        <label>
          <input type="radio" name="modelSelect" value="1" checked />
          1: Simulation
        </label>
        <label>
          <input type="radio" name="modelSelect" value="2" />
          2: Laerbeekbos (Brussels)
        </label>
        <label>
          <input type="radio" name="modelSelect" value="3" />
          3: Kaai (Ehb)
        </label>
      </div>
    </fieldset>

    <div class="compassWrap">
      <canvas id="compass" width="140" height="140"></canvas>
      <div class="row">Heading: <span id="heading">--</span>&deg;</div>
    </div>

    <div class="controls">
      <button id="btn">Start</button>
      <button id="switchCam" disabled>Switch Camera</button>

      <label class="fpsControl" for="fpsSelect">
        FPS
        <select id="fpsSelect">
          <option value="1">1</option>
          <option value="2" selected>2</option>
          <option value="3">3</option>
          <option value="4">4</option>
          <option value="5">5</option>
          <option value="6">6</option>
          <option value="7">7</option>
          <option value="8">8</option>
          <option value="9">9</option>
          <option value="10">10</option>
        </select>
      </label>

      <label class="fpsControl" for="forwardDegSelect">
        Forward (deg)
        <select id="forwardDegSelect">
          <option value="8" selected>8</option>
          <option value="9">9</option>
          <option value="10">10</option>
          <option value="11">11</option>
          <option value="12">12</option>
          <option value="13">13</option>
          <option value="14">14</option>
          <option value="15">15</option>
          <option value="16">16</option>
        </select>
      </label>

      <label class="fpsControl" for="detectionConfidenceSelect">
        Detection confidence
        <select id="detectionConfidenceSelect">
          <option value="0.1">0.1</option>
          <option value="0.2">0.2</option>
          <option value="0.3">0.3</option>
          <option value="0.4">0.4</option>
          <option value="0.5" selected>0.5</option>
          <option value="0.6">0.6</option>
          <option value="0.7">0.7</option>
          <option value="0.8">0.8</option>
          <option value="0.9">0.9</option>
        </select>
      </label>
    </div>
  </div>
</div>

<video id="video" autoplay playsinline></video>
<canvas id="cap"></canvas>

<script>
(() => {
  const SIGNALING_SERVER = <?php echo json_encode($wsUrl, JSON_UNESCAPED_SLASHES); ?>;
  const BEARER_TOKEN = <?php echo json_encode($bearerToken); ?>;

  const TARGET_W = 640, TARGET_H = 480;
  const JPEG_QUALITY = 0.70;

  const statusEl = document.getElementById("status");
  const sentEl = document.getElementById("sent");
  const kbpsEl = document.getElementById("kbps");
  const errsEl = document.getElementById("errs");
  const latencyEl = document.getElementById("latency");
  const headingEl = document.getElementById("heading");
  const btn = document.getElementById("btn");
  const switchCamBtn = document.getElementById("switchCam");
  const fpsSelect = document.getElementById("fpsSelect");
  const forwardDegSelect = document.getElementById("forwardDegSelect");
  const detectionConfidenceSelect = document.getElementById("detectionConfidenceSelect");
  const modelInputs = document.querySelectorAll('input[name="modelSelect"]');

  const video = document.getElementById("video");
  const cap = document.getElementById("cap");
  const ctx = cap.getContext("2d", { alpha: false });
  const compass = document.getElementById("compass");
  const compCtx = compass.getContext("2d");

  let ws = null;
  let stream = null;
  let activeVideoDeviceId = null;
  let availableVideoInputs = [];
  let timer = null;
  let sentFrames = 0;
  let errors = 0;
  let latestHeading = null;
  let nextFrameId = 1;
  const sentAtByFrameId = new Map();
  let geoWatchId = null;
  let latestLatitude = null;
  let latestLongitude = null;
  let fps = Number(fpsSelect?.value) || 2;
  let forwardDeg = Number(forwardDegSelect?.value) || 8;
  let detectionConfidence = Number(detectionConfidenceSelect?.value) || 0.5;
  let currentSessionId = null;
  let isAuthenticated = false;
  let authStarted = false;

  let bytesSince = 0;
  let lastRateT = performance.now();

  const SOUND_MAP = {
    left: "audio/left.mp3",
    right: "audio/right.mp3",
    forward: "audio/forward.mp3",
    started: "audio/application_started.mp3",
    be_carefull: "audio/be_carefull.mp3",
    beep: "audio/beep.wav"
  };

  const player = new Audio();
  player.preload = "auto";

  const preloaded = {};
  for (const [k, url] of Object.entries(SOUND_MAP)) {
    const a = new Audio(url);
    a.preload = "auto";
    preloaded[k] = a;
  }

  let audioEnabled = false;
  let lastCmd = null;
  let lastCmdAt = 0;
  let noDetectionSince = null;
  let noDetectionWarned = false;
  let targetHeading = null;

  const COOLDOWN_MS = 5000;

  function angleDiffDeg(current, target) {
    return (current - target + 540) % 360 - 180;
  }

  function headingToCmd(heading) {
    if (targetHeading === null) return null;
    const d = angleDiffDeg(heading, targetHeading);
    if (Math.abs(d) <= forwardDeg) return "forward";
    return d > 0 ? "left" : "right";
  }

  async function playCmd(cmd) {
    if (!audioEnabled) return;
    if (!cmd || !SOUND_MAP[cmd]) return;

    const now = performance.now();
    if (cmd === lastCmd && (now - lastCmdAt) < COOLDOWN_MS) return;

    lastCmd = cmd;
    lastCmdAt = now;

    try {
      player.pause();
      player.currentTime = 0;
      player.src = SOUND_MAP[cmd];
      await player.play();
    } catch (e) {
      console.log("Audio play blocked/failed:", e);
    }
  }

  function drawArrow(headingDeg) {
    const w = compass.width;
    const h = compass.height;
    const cx = w / 2;
    const cy = h / 2;

    compCtx.clearRect(0, 0, w, h);

    compCtx.beginPath();
    compCtx.arc(cx, cy, 62, 0, Math.PI * 2);
    compCtx.strokeStyle = "rgba(255,255,255,0.35)";
    compCtx.lineWidth = 2;
    compCtx.stroke();

    compCtx.fillStyle = "rgba(255,255,255,0.7)";
    compCtx.font = "12px system-ui";
    compCtx.textAlign = "center";
    compCtx.fillText("Forward", cx, 14);

    if (typeof headingDeg !== "number" || Number.isNaN(headingDeg)) return;

    const angleRad = (-headingDeg * Math.PI) / 180;
    compCtx.save();
    compCtx.translate(cx, cy);
    compCtx.rotate(angleRad);

    compCtx.beginPath();
    compCtx.moveTo(-8, -3);
    compCtx.lineTo(40, -3);
    compCtx.lineTo(40, -10);
    compCtx.lineTo(56, 0);
    compCtx.lineTo(40, 10);
    compCtx.lineTo(40, 3);
    compCtx.lineTo(-8, 3);
    compCtx.closePath();
    compCtx.fillStyle = "#ff3b30";
    compCtx.fill();
    compCtx.restore();
  }

  function normalizeHeading(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    return ((n % 360) + 360) % 360;
  }

  function createSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return `sess-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }

  function setStatus(s) {
    statusEl.textContent = s;
  }

  function getSelectedModel() {
    const selected = Array.from(modelInputs).find((input) => input.checked);
    return selected?.value || "2";
  }

  function getIntervalMs() {
    return Math.round(1000 / fps);
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

  function incErr() {
    errors++;
    errsEl.textContent = String(errors);
  }

  function startLocationTracking() {
    if (!("geolocation" in navigator) || geoWatchId !== null) return;
    try {
      geoWatchId = navigator.geolocation.watchPosition(
        (pos) => {
          const lat = Number(pos?.coords?.latitude);
          const lon = Number(pos?.coords?.longitude);
          latestLatitude = Number.isFinite(lat) ? lat : null;
          latestLongitude = Number.isFinite(lon) ? lon : null;
        },
        (err) => {
          console.warn("Geolocation unavailable:", err);
        },
        { enableHighAccuracy: true, maximumAge: 3000, timeout: 10000 }
      );
    } catch (err) {
      console.warn("Failed to start geolocation:", err);
    }
  }

  function stopLocationTracking() {
    if (!("geolocation" in navigator)) return;
    if (geoWatchId !== null) {
      try { navigator.geolocation.clearWatch(geoWatchId); } catch {}
      geoWatchId = null;
    }
    latestLatitude = null;
    latestLongitude = null;
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

    audioEnabled = true;
    targetHeading = null;
    lastCmd = null;
    lastCmdAt = 0;
    noDetectionSince = null;
    noDetectionWarned = false;
    playCmd("started");

    restartCaptureTimer();
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
    startLocationTracking();
    currentSessionId = createSessionId();
    isAuthenticated = false;
    authStarted = false;

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

      ws.onmessage = (msg) => {
        let heading = null;
        let frameId = null;
        let incomingSessionId = null;

        if (typeof msg.data === "string") {
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
              if (!isAuthenticated) {
                beginAuthenticatedStreaming();
              }
              return;
            }

            incomingSessionId = payload?.sessionId ?? payload?.session_id ?? null;

            if (currentSessionId && incomingSessionId && incomingSessionId !== currentSessionId) {
              return;
            }

            if (currentSessionId && !incomingSessionId) {
              return;
            }

            heading = payload?.heading;
            frameId = payload?.frame_id;
          } catch {
            return;
          }
        } else {
          return;
        }

        const normalized = normalizeHeading(heading);
        if (normalized !== null) {
          latestHeading = normalized;
          headingEl.textContent = normalized.toFixed(1);
          drawArrow(latestHeading);

          const isNoDetection = normalized === 90;
          if (isNoDetection) {
            if (noDetectionSince === null) {
              noDetectionSince = performance.now();
              noDetectionWarned = false;
              lastCmd = null;
              lastCmdAt = 0;
            } else if (!noDetectionWarned && (performance.now() - noDetectionSince) >= 2000) {
              noDetectionWarned = true;
              playCmd("beep");
            }
          } else {
            noDetectionSince = null;
            noDetectionWarned = false;
            if (targetHeading === null) {
              targetHeading = latestHeading;
            } else {
              const cmd = headingToCmd(latestHeading);
              playCmd(cmd);
            }
          }
        }

        if (frameId !== null && frameId !== undefined) {
          const id = String(frameId);
          const sentAt = sentAtByFrameId.get(id);
          if (typeof sentAt === "number") {
            const latencyMs = performance.now() - sentAt;
            latencyEl.textContent = latencyMs.toFixed(1);
            sentAtByFrameId.delete(id);
          }
        }
      };
    } catch (e) {
      incErr();
      setStatus("WS connect failed");
      console.error(e);
      stop(true);
      return;
    }
  }

  function stop(allowButton = true) {
    if (timer) { clearInterval(timer); timer = null; }
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
    stopLocationTracking();

    audioEnabled = false;
    targetHeading = null;
    noDetectionSince = null;
    noDetectionWarned = false;
    lastCmd = null;
    latestHeading = null;
    headingEl.textContent = "--";
    latencyEl.textContent = "--";

    try {
      player.pause();
      player.currentTime = 0;
    } catch {}

    if (allowButton) {
      btn.disabled = false;
      btn.textContent = "Start";
      setStatus("Idle");
    }
  }

  function updateRate(bytesJustSent) {
    bytesSince += bytesJustSent;
    const now = performance.now();
    const dt = now - lastRateT;
    if (dt >= 1000) {
      const kbitsPerSec = (bytesSince * 8) / dt;
      kbpsEl.textContent = kbitsPerSec.toFixed(1);
      bytesSince = 0;
      lastRateT = now;
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
          sessionId: currentSessionId,
          latitude: latestLatitude,
          longitude: latestLongitude,
          lastlatency: latencyEl.textContent === "--" ? null : Number(latencyEl.textContent),
          model: getSelectedModel(),
          detection_confidence: detectionConfidence
        }));

        sentAtByFrameId.set(frameId, performance.now());
        ws.send(buf);

        sentFrames++;
        sentEl.textContent = String(sentFrames);
        updateRate(buf.byteLength);

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

  fpsSelect.addEventListener("change", () => {
    const nextFps = Number(fpsSelect.value);
    if (!Number.isInteger(nextFps) || nextFps < 1 || nextFps > 10) {
      fps = 2;
      fpsSelect.value = "2";
    } else {
      fps = nextFps;
    }
    restartCaptureTimer();
  });

  forwardDegSelect.addEventListener("change", () => {
    const nextForwardDeg = Number(forwardDegSelect.value);
    if (!Number.isInteger(nextForwardDeg) || nextForwardDeg < 8 || nextForwardDeg > 16) {
      forwardDeg = 8;
      forwardDegSelect.value = "8";
    } else {
      forwardDeg = nextForwardDeg;
    }
  });

  detectionConfidenceSelect.addEventListener("change", () => {
    const nextDetectionConfidence = Number(detectionConfidenceSelect.value);
    if (!Number.isFinite(nextDetectionConfidence) || nextDetectionConfidence < 0.1 || nextDetectionConfidence > 0.9) {
      detectionConfidence = 0.5;
      detectionConfidenceSelect.value = "0.5";
    } else {
      detectionConfidence = Number(nextDetectionConfidence.toFixed(1));
    }
  });

  drawArrow(null);
})();
</script>
<?php endif; ?>
</main>
</body>
</html>