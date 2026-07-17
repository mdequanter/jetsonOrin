<?php
require_once __DIR__ . DIRECTORY_SEPARATOR . 'auth.php';
require_once __DIR__ . DIRECTORY_SEPARATOR . 'menu.php';
require_once __DIR__ . DIRECTORY_SEPARATOR . 'route_repository.php';
auth_start_session();

$bearerToken = 'LTddk_ptxQX-omdw5B5rfpniA2wB-19KBxFaKuODMzw';
$wsUrl = 'wss://signaling.ehb.be';
$room = '/ws/pathnavigation';
$currentUser = auth_current_user();
$currentUserId = is_array($currentUser) ? (int)($currentUser['id'] ?? 0) : null;

// Default navigation preferences (match the historical defaults for anonymous visitors).
$userPreferences = [
    'guiding_beeps'    => true,
    'vibration'        => false,
    'mapview'          => true,
    'cameraview'       => true,
    'extra_details'        => true,
    'preferred_model'      => 'laerbeekbos',
    'update_frequency'     => 2.0,
    'model_confidence'     => 0.5,
    'course_tolerance_deg' => 5.0,
];
if ($currentUserId) {
    try {
        $prefsPdo = auth_db();
        auth_ensure_preferences_table($prefsPdo);
        $userPreferences = auth_get_preferences($prefsPdo, $currentUserId);
    } catch (Throwable $e) {
        error_log('followpath: could not load user preferences: ' . $e->getMessage());
    }
}

function jsonResponse(array $payload, int $statusCode = 200): void {
    http_response_code($statusCode);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($payload, JSON_UNESCAPED_SLASHES | JSON_PRETTY_PRINT);
    exit;
}

function h(?string $value): string {
    return htmlspecialchars((string)$value, ENT_QUOTES, 'UTF-8');
}

function slugifyRouteName(string $value): string {
    $value = trim(strtolower($value));
    $value = preg_replace('/[^a-z0-9]+/', '-', $value) ?? '';
    $value = trim($value, '-');
    return $value !== '' ? $value : 'path';
}

function listSavedPaths(PDO $pdo, ?int $userId): array {
    return route_repository_list_visible_to_user($pdo, $userId);
}

function loadSavedPath(PDO $pdo, string $slug, ?int $userId): ?array {
    $safeSlug = slugifyRouteName($slug);
    return route_repository_load_visible_document($pdo, $safeSlug, $userId);
}

if ($_SERVER['REQUEST_METHOD'] === 'GET' && isset($_GET['action'])) {
    if ($_GET['action'] === 'list_paths') {
        try {
            jsonResponse(['ok' => true, 'paths' => listSavedPaths(auth_db(), $currentUserId)]);
        } catch (Throwable $error) {
            jsonResponse(['ok' => false, 'error' => $error->getMessage()], 500);
        }
    }

    if ($_GET['action'] === 'load_path') {
        $slug = (string)($_GET['slug'] ?? '');
        if ($slug === '') {
            jsonResponse(['ok' => false, 'error' => 'Missing path slug.'], 400);
        }

        try {
            $path = loadSavedPath(auth_db(), $slug, $currentUserId);
        } catch (Throwable $error) {
            jsonResponse(['ok' => false, 'error' => $error->getMessage()], 500);
        }
        if ($path === null) {
            jsonResponse(['ok' => false, 'error' => 'Path not found.'], 404);
        }

        jsonResponse(['ok' => true, 'path' => $path]);
    }

    jsonResponse(['ok' => false, 'error' => 'Unknown action.'], 404);
}
?><!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stay On Trails - Vrij wandelen</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root{
      --focus:#22d3ee;
      --menu-bg:rgba(15,23,42,.92);
      --menu-border:rgba(255,255,255,.18);
      --accent:#facc15;
      --accent-ink:#111827;
      --line:rgba(255,255,255,.12);
      --muted:#cbd5e1;
      --ok:#34d399;
      --warn:#fb7185;
    }
    *{box-sizing:border-box}
    html,body{margin:0;min-height:100%;background:#020617;color:#fff;font-family:Arial,Helvetica,sans-serif}
    a:focus-visible,button:focus-visible,select:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
    .topbar{position:sticky;top:0;z-index:20;background:var(--menu-bg);border-bottom:1px solid var(--menu-border);backdrop-filter:blur(10px)}
    .topbar-inner{max-width:1420px;margin:0 auto;padding:12px 18px;display:flex;align-items:center;justify-content:flex-start;gap:12px}
    .brand{margin:0;font-size:18px;font-weight:700}
    .menu{list-style:none;margin:0;padding:0;display:flex;gap:10px;flex-wrap:wrap}
    .menu a{display:inline-block;color:#fff;text-decoration:none;font-weight:700;padding:8px 10px;border-radius:8px}
    .menu a:hover{background:rgba(255,255,255,.1)}
    .menu .cta{background:var(--accent);color:var(--accent-ink)}
    .layout{max-width:760px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:14px}
    .panel{background:rgba(15,23,42,.84);border:1px solid var(--line);border-radius:18px;box-shadow:0 18px 40px rgba(0,0,0,.26)}
    .mapPanel{padding:14px}
    .sidePanel{padding:16px;display:flex;flex-direction:column;gap:14px}
    .toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    .buttonStack{display:flex;flex-direction:column;gap:10px}
    .buttonStack .btn{width:100%;text-align:center;display:block}
    .walkRow{display:flex;gap:10px}
    .walkRow .btn{flex:1 1 0;width:auto}
    .btn[aria-pressed="true"]{background:var(--accent);color:var(--accent-ink);border-color:rgba(250,204,21,.35)}
    .srOnly{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0 0 0 0);clip-path:inset(50%);white-space:nowrap;border:0}
    .field{display:flex;flex-direction:column;gap:6px}
    .field label,.sectionTitle{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
    .field select{width:100%;padding:11px 12px;border-radius:12px;border:1px solid var(--line);background:#020617;color:#fff;font-size:14px}
    .btn{padding:11px 14px;border-radius:12px;border:1px solid var(--line);background:#1e293b;color:#fff;font-weight:700;cursor:pointer}
    .btn:hover{background:#334155}
    .btnPrimary{background:var(--accent);color:var(--accent-ink);border-color:rgba(250,204,21,.35)}
    .btnPrimary:hover{background:#fde047}
    .btnSmall{padding:8px 10px;font-size:12px}
    .mapWrap{position:relative}
    #map{height:48vh;min-height:300px;width:100%;border-radius:16px;overflow:hidden;border:1px solid var(--line)}
    .detailsBlock{border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.04);padding:6px 14px}
    .detailsBlock summary{cursor:pointer;font-weight:700;padding:10px 0;list-style:none}
    .detailsBlock summary::-webkit-details-marker{display:none}
    .detailsBlock summary::before{content:"▸ ";color:var(--muted)}
    .detailsBlock[open] summary::before{content:"▾ "}
    .detailsBody{display:flex;flex-direction:column;gap:12px;padding:4px 0 12px}
    .help{margin:10px 0 0;color:var(--muted);font-size:14px;line-height:1.5}
    .status{padding:10px 12px;border-radius:12px;border:none var(--line);font-size:14px;color:var(--muted)}
    .status.ok{color:#d1fae5;border-color:rgba(52,211,153,.35);background:rgba(16,185,129,.08)}
    .status.warn{color:#ffe4e6;border-color:rgba(251,113,133,.35);background:rgba(244,63,94,.08)}
    .card{padding:14px;border-radius:16px;background:rgba(255,255,255,.04);border:1px solid var(--line)}
    .bigInstruction{font-size:28px;line-height:1.2;font-weight:700}
    .muted{color:var(--muted);font-size:14px;line-height:1.5}
    .stats{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .stat{padding:12px;border-radius:14px;background:#0b1220;border:1px solid var(--line)}
    .statLabel{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:8px}
    .statValue{font-size:20px;font-weight:700}
    .trailDirectionValue{font-size:24px;font-weight:700}
    .mapHeadingOverlay{position:absolute;left:16px;bottom:16px;z-index:500;display:flex;flex-direction:column;align-items:center;gap:8px;padding:12px 14px;border-radius:18px;background:rgba(2,6,23,.22);box-shadow:0 18px 40px rgba(0,0,0,.28);pointer-events:none}
    .mapHeadingValue{font-size:22px;font-weight:700}
    .mapCameraOverlay{position:relative;overflow:hidden;width:100%;aspect-ratio:4/3;border-radius:16px;border:1px solid var(--line);background:#020617;box-shadow:0 18px 40px rgba(0,0,0,.28)}
    .arSwitchLink{position:absolute;top:24px;right:24px;z-index:520;display:inline-flex;align-items:center;justify-content:center;width:44px;height:34px;border-radius:12px;border:1px solid rgba(255,255,255,.26);background:rgba(2,6,23,.78);color:#fff;text-decoration:none;box-shadow:0 12px 28px rgba(0,0,0,.34);backdrop-filter:blur(8px)}
    .arSwitchLink:hover{background:rgba(15,23,42,.92)}
    .arSwitchLink svg{display:block;width:28px;height:18px}
    .trailVideoBg{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;z-index:0;background:#000;opacity:0}
    .trailPreviewBg{position:absolute;inset:0;width:100%;height:100%;z-index:0;background:#000}
    .trailMaskOverlay{position:absolute;inset:0;width:100%;height:100%;z-index:1;pointer-events:none}
    .hidden{display:none !important}
    .compassWrap{display:flex;flex-direction:column;align-items:center;gap:6px}
    #mapCompass{width:110px;height:110px;display:block}
    .list{display:flex;flex-direction:column;gap:10px;max-height:34vh;overflow:auto;padding-right:4px}
    .pointItem{padding:12px;border-radius:14px;border:1px solid var(--line);background:#0b1220}
    .pointItem.active{border-color:rgba(34,211,238,.55);background:#0f1a2c}
    .pointTitle{font-size:13px;font-weight:700;margin-bottom:6px}
    .pointText,.pointMeta{font-size:13px;line-height:1.45}
    .pointMeta{color:var(--muted)}
    .leaflet-div-icon{background:transparent;border:0}
    .personMarkerInner{position:relative;width:22px;height:30px}
    .personMarkerInner::before{content:"";position:absolute;left:6px;top:0;width:10px;height:10px;border-radius:50%;background:#facc15;box-shadow:0 0 0 2px rgba(17,24,39,.92)}
    .personMarkerInner::after{content:"";position:absolute;left:4px;top:11px;width:14px;height:16px;border-radius:8px 8px 6px 6px;background:#facc15;box-shadow:0 0 0 2px rgba(17,24,39,.92)}
    .waypointBubble{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#1d4ed8;color:#fff;font-weight:700;border:2px solid rgba(15,23,42,.92)}
    .waypointBubble.active{background:#22d3ee;color:#082f49}
    .waypointBubble.done{background:#16a34a;color:#ecfdf5}
    @media (max-width: 1100px){
      #map{height:46vh;min-height:300px}
      .mapHeadingOverlay{left:12px;bottom:12px}
      .arSwitchLink{top:20px;right:20px}
    }
    @media (max-width: 720px){
      .layout{padding:12px}
      .stats{grid-template-columns:1fr}
      .mapHeadingOverlay{left:12px;right:12px;bottom:12px}
      .arSwitchLink{top:18px;right:18px;width:40px;height:32px}
      .mapHeadingValue{font-size:20px}
    }
  </style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <?php sot_render_menu(); ?>
  </div>
</header>

<main class="layout">
  <!-- Alle knoppen, volledige breedte, Start en Stop eerst -->
  <div class="buttonStack">
    <button id="startBtn" class="btn btnPrimary" type="button">Start wandelen</button>
    <button id="stopBtn" class="btn hidden" type="button">Stop</button>
    <div class="walkRow">
      <button id="walkLeftBtn" class="btn" type="button" aria-pressed="false" aria-label="Links wandelen">Links</button>
      <button id="walkCenterBtn" class="btn" type="button" aria-pressed="true" aria-label="Midden wandelen">Midden</button>
      <button id="walkRightBtn" class="btn" type="button" aria-pressed="false" aria-label="Rechts wandelen">Rechts</button>
    </div>
    <button id="satelliteToggleBtn" class="btn" type="button">Satelliet tonen</button>
    <button id="saveGpxBtn" class="btn hidden" type="button">GPX opslaan</button>
  </div>

  <!-- Schermlezer-aankondiging (TalkBack leest wijzigingen hier voor). -->
  <div id="a11yAnnounce" class="srOnly" aria-live="assertive"></div>

  <!-- Modelkeuze: verborgen — het model komt uit de gebruikersvoorkeuren. -->
  <div class="field hidden" id="modelField" aria-hidden="true">
    <label for="modelSelect">Model</label>
    <select id="modelSelect">
      <option value="unrealsim">UnrealSim</option>
      <option value="laerbeekbos" selected>Laarbeekbos</option>
      <option value="kaai">Kaai</option>
      <option value="denham">Denham</option>
    </select>
  </div>

  <!-- Verborgen knoppen, behouden voor de gedeelde logica (niet gebruikt bij vrij wandelen) -->
  <div class="hidden" aria-hidden="true">
    <button id="prevWaypointBtn" type="button"></button>
    <button id="nextWaypointBtn" type="button"></button>
    <button id="helpBtn" type="button" disabled></button>
    <button id="repeatBtn" type="button"></button>
    <button id="headingSpeechToggleBtn" type="button"></button>
  </div>

  <!-- Kaart -->
  <div class="mapWrap" id="mapWrap" aria-hidden="true">
    <div id="map" aria-label="Wandelkaart"></div>
    <div id="mapHeadingOverlay" class="mapHeadingOverlay">
      <div class="compassWrap">
        <canvas id="mapCompass" width="110" height="110"></canvas>
      </div>
      <div id="mapTrailDirectionValue" class="mapHeadingValue">--</div>
      <div id="mapTrailDirectionMeta" class="muted">Segmentatiebegeleiding inactief.</div>
    </div>
  </div>

  <!-- Camerabeeld -->
  <div id="mapCameraOverlay" class="mapCameraOverlay" aria-hidden="true">
    <video id="trailVideo" class="trailVideoBg" autoplay playsinline muted></video>
    <canvas id="trailPreview" class="trailPreviewBg"></canvas>
    <canvas id="trailMaskOverlay" class="trailMaskOverlay"></canvas>
  </div>

  <!-- Begeleidingstekst -->
  <div class="card" aria-hidden="true">
    <div id="currentInstruction" class="bigInstruction">Vrij wandelen</div>
    <div id="currentInstructionMeta" class="muted" style="margin-top:10px">Kies een model en start. De begeleiding houdt je in het midden van het pad.</div>
  </div>

  <!-- Hulp op afstand (enkel zichtbaar bij een actieve sessie) -->
  <div class="card hidden" id="remoteAssistantCard" aria-hidden="true">
    <div class="sectionTitle">Hulp op afstand</div>
    <div id="helperMessageValue" class="bigInstruction" style="margin-top:12px;font-size:22px">Geen bericht.</div>
    <div id="helperMessageMeta" class="muted" style="margin-top:10px">Berichten van de hulp op afstand verschijnen hier.</div>
  </div>

  <div id="status" class="status" aria-hidden="true"></div>

  <!-- Details -->
  <details class="detailsBlock" aria-hidden="true">
    <summary>Details</summary>
    <div class="detailsBody">
      <div class="card">
        <div class="sectionTitle">GPS-status</div>
        <div class="stats" style="margin-top:12px">
          <div class="stat"><div class="statLabel">Breedtegraad</div><div id="latValue" class="statValue">--</div></div>
          <div class="stat"><div class="statLabel">Lengtegraad</div><div id="lonValue" class="statValue">--</div></div>
          <div class="stat"><div class="statLabel">Nauwkeurigheid</div><div id="accuracyValue" class="statValue">--</div></div>
          <div class="stat"><div class="statLabel">Opgenomen punten</div><div id="distanceValue" class="statValue">0</div></div>
        </div>
        <div id="gpsStatus" class="muted" style="margin-top:12px">GPS niet gestart.</div>
      </div>
      <div class="card">
        <div class="sectionTitle">Telefoonoriëntatie</div>
        <div class="stats" style="margin-top:12px">
          <div class="stat"><div class="statLabel">Kantelhoek</div><div id="downAngleValue" class="statValue">--</div></div>
          <div class="stat"><div class="statLabel">Waterpas</div><div id="levelAngleValue" class="statValue">--</div></div>
          <div class="stat"><div class="statLabel">Staand</div><div id="portraitValue" class="statValue">--</div></div>
        </div>
        <div id="orientationStatus" class="muted" style="margin-top:12px">Kantelsensor niet gestart.</div>
      </div>
      <div class="card">
        <div class="sectionTitle">Streaming</div>
        <div class="stats" style="margin-top:12px">
          <div class="stat"><div class="statLabel">Frames verzonden</div><div id="sentFramesValue" class="statValue">0</div></div>
          <div class="stat"><div class="statLabel">Verzendsnelheid</div><div id="sendRateValue" class="statValue">0.0 fps</div></div>
          <div class="stat"><div class="statLabel">Vertraging</div><div id="latencyValue" class="statValue">--</div></div>
        </div>
        <div id="streamMeta" class="muted" style="margin-top:12px">Ingesteld interval: 1.0 s per frame.</div>
      </div>
    </div>
  </details>

  <!-- Verborgen elementen, behouden voor de bestaande route- en sessielogica -->
  <select id="savedPaths" class="hidden" aria-hidden="true" tabindex="-1"><option value=""></option></select>
  <span id="sessionIdValue" class="hidden"></span>
  <span id="sessionIdMeta" class="hidden"></span>
  <button id="copySessionBtn" class="hidden" type="button" tabindex="-1" aria-hidden="true"></button>
</main>

<canvas id="trailCap" style="display:none"></canvas>

<script>
  const API_URL = <?php echo json_encode(basename(__FILE__), JSON_UNESCAPED_SLASHES); ?>;
  const SIGNALING_SERVER = <?php echo json_encode($wsUrl, JSON_UNESCAPED_SLASHES); ?>;
  const BEARER_TOKEN = <?php echo json_encode($bearerToken); ?>;
  const SIGNALING_ROOM = <?php echo json_encode($room, JSON_UNESCAPED_SLASHES); ?>;
  const DEFAULT_CENTER = [50.8503, 4.3517];
  const LAST_SELECTED_PATH_STORAGE_KEY = "stayontrails.lastSelectedPathSlug";
  const DEFAULT_ARRIVAL_RADIUS_METERS = 3;
  const ALLOWED_ARRIVAL_RADIUS_METERS = [1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30];
  const OFF_ROUTE_WARNING_METERS = 3000;
  const TARGET_W = 640;
  const TARGET_H = 480;
  const JPEG_QUALITY = 0.70;
  const DEFAULT_LANGUAGE = "nl-BE";
  const DEFAULT_HEADING_FEEDBACK_FPS = 2.0;
  const DEFAULT_MODEL = "unrealsim";
  const DEFAULT_MODEL_CONFIDENCE = 0.5;
  const LATENCY_WARNING_THRESHOLD_MS = 500;
  const DOWN_ANGLE_TARGET_DEG = 90;
  const DOWN_ANGLE_TOLERANCE_DEG = 20;
  const LEVEL_TOLERANCE_DEG = 5;
  // Toegestane koersafwijking rond 90° (uit de gebruikersvoorkeuren): binnen deze marge geldt "rechtdoor".
  const COURSE_TOLERANCE_DEG = <?php echo json_encode($userPreferences['course_tolerance_deg'] ?? 5); ?>;
  // De trilling/tonen reageren 5° later dan de gesproken/getoonde richting (historisch verschil behouden).
  const HAPTIC_COURSE_TOLERANCE_DEG = COURSE_TOLERANCE_DEG + 5;
  const ARUCO_MARKER_SPEECH_COOLDOWN_MS = 60000;
  const ARUCO_DISTANCE_SPEECH_THRESHOLD_METERS = 2.0;

  const topbarEl = document.querySelector(".topbar");
  const savedPathsEl = document.getElementById("savedPaths");
  const startBtn = document.getElementById("startBtn");
  const helpBtn = document.getElementById("helpBtn");
  const repeatBtn = document.getElementById("repeatBtn");
  const headingSpeechToggleBtn = document.getElementById("headingSpeechToggleBtn");
  const satelliteToggleBtn = document.getElementById("satelliteToggleBtn");
  const walkLeftBtn = document.getElementById("walkLeftBtn");
  const walkCenterBtn = document.getElementById("walkCenterBtn");
  const walkRightBtn = document.getElementById("walkRightBtn");
  const a11yAnnounceEl = document.getElementById("a11yAnnounce");
  const stopBtn = document.getElementById("stopBtn");
  const prevWaypointBtn = document.getElementById("prevWaypointBtn");
  const nextWaypointBtn = document.getElementById("nextWaypointBtn");
  const modelSelectEl = document.getElementById("modelSelect");
  const modelFieldEl = document.getElementById("modelField");
  const saveGpxBtn = document.getElementById("saveGpxBtn");
  const arSwitchLinkEl = document.getElementById("arSwitchLink");
  const statusEl = document.getElementById("status");
  const gpsStatusEl = document.getElementById("gpsStatus");
  const latValueEl = document.getElementById("latValue");
  const lonValueEl = document.getElementById("lonValue");
  const accuracyValueEl = document.getElementById("accuracyValue");
  const distanceValueEl = document.getElementById("distanceValue");
  const downAngleValueEl = document.getElementById("downAngleValue");
  const levelAngleValueEl = document.getElementById("levelAngleValue");
  const portraitValueEl = document.getElementById("portraitValue");
  const orientationStatusEl = document.getElementById("orientationStatus");
  const currentInstructionEl = document.getElementById("currentInstruction");
  const currentInstructionMetaEl = document.getElementById("currentInstructionMeta");
  const helperMessageValueEl = document.getElementById("helperMessageValue");
  const helperMessageMetaEl = document.getElementById("helperMessageMeta");
  const sessionIdValueEl = document.getElementById("sessionIdValue");
  const sessionIdMetaEl = document.getElementById("sessionIdMeta");
  const copySessionBtn = document.getElementById("copySessionBtn");
  const remoteAssistantCardEl = document.getElementById("remoteAssistantCard");
  const mapWrapEl = document.getElementById("mapWrap");
  const mapHeadingOverlayEl = document.getElementById("mapHeadingOverlay");
  const mapCameraOverlayEl = document.getElementById("mapCameraOverlay");
  const mapTrailDirectionValueEl = document.getElementById("mapTrailDirectionValue");
  const mapTrailDirectionMetaEl = document.getElementById("mapTrailDirectionMeta");
  const sentFramesValueEl = document.getElementById("sentFramesValue");
  const sendRateValueEl = document.getElementById("sendRateValue");
  const latencyValueEl = document.getElementById("latencyValue");
  const streamMetaEl = document.getElementById("streamMeta");
  const trailVideoEl = document.getElementById("trailVideo");
  const trailPreviewEl = document.getElementById("trailPreview");
  const trailPreviewCtx = trailPreviewEl.getContext("2d", { alpha: false });
  const trailMaskOverlayEl = document.getElementById("trailMaskOverlay");
  const trailMaskOverlayCtx = trailMaskOverlayEl.getContext("2d");
  const trailCapEl = document.getElementById("trailCap");
  const trailCapCtx = trailCapEl.getContext("2d", { alpha: false });
  const mapCompass = document.getElementById("mapCompass");
  const mapCompCtx = mapCompass.getContext("2d");

  const map = L.map("map").setView(DEFAULT_CENTER, 16);
  const streetLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);
  const satelliteLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles &copy; Esri"
  });

  const personIcon = L.divIcon({
    className: "",
    html: '<div class="personMarkerInner" aria-hidden="true"></div>',
    iconSize: [22, 30],
    iconAnchor: [11, 26]
  });

  let currentLocationMarker = L.marker(DEFAULT_CENTER, {
    icon: personIcon,
    title: "Huidige locatie"
  }).addTo(map);
  let pathLine = null;
  let waypointMarkers = [];
  let geoWatchId = null;
  let latestLatitude = null;
  let latestLongitude = null;
  let latestAccuracy = null;
  let latestPitchDeg = null;
  let latestRollDeg = null;
  let orientationTrackingActive = false;
  let orientationListenerAttached = false;
  let lastDownAngleWarningAtMs = 0;
  let lastLevelWarningAtMs = 0;
  let currentPath = null;
  let currentLanguage = DEFAULT_LANGUAGE;
  let currentModel = DEFAULT_MODEL;
  let currentModelConfidence = DEFAULT_MODEL_CONFIDENCE;
  let currentReturnMasks = true;
  let recordedTrack = [];
  let currentSendMQTT = false;
  let headingFeedbackFps = DEFAULT_HEADING_FEEDBACK_FPS;
  let sendIntervalMs = Math.max(100, Math.round(1000 / DEFAULT_HEADING_FEEDBACK_FPS));
  let routePoints = [];
  let arrivalRadiusMeters = DEFAULT_ARRIVAL_RADIUS_METERS;
  let activePointIndex = 0;
  let lastSpokenPointId = null;
  let walkingActive = false;
  // Instelbare referentie-heading: "Meer links"/"Meer rechts" verschuiven het pad-midden weg van 90°.
  let referenceHeading = 90;
  let ws = null;
  let stream = null;
  let activeVideoDeviceId = null;
  let timer = null;
  let nextFrameId = 1;
  let currentSessionId = null;
  let helperEnabled = false;
  let isAuthenticated = false;
  let authStarted = false;
  let latestHeading = null;
  let latestMarkerHeading = null;
  let latestMarkerId = null;
  let consecutiveNoTrackingHeadingUpdates = 0;
  let lastSpokenDirectionKey = null;
  let lastHapticDirectionKey = null;
  let lastHapticAtMs = 0;
  let lastTurnHapticDirectionKey = null;
  let lastTurnHapticCount = 0;
  let stereoAudioContext = null;
  let activeSpeechKind = null;
  let lastInstructionText = "";
  let offRouteWarningActive = false;
  let lastHelperMessageText = "";
  const USER_PREFERENCES = <?php echo json_encode($userPreferences, JSON_UNESCAPED_SLASHES); ?>;
  const mapViewEnabled = USER_PREFERENCES.mapview !== false;
  const cameraViewEnabled = USER_PREFERENCES.cameraview !== false;
  // USER_PREFERENCES.extra_details is reserved for an upcoming feature
  // (extra info field + per-waypoint details) and is not wired up yet.
  const extraDetailsEnabled = USER_PREFERENCES.extra_details === true;
  let headingSpeechEnabled = true;
  let beepsEnabled = USER_PREFERENCES.guiding_beeps !== false;
  let hapticsEnabled = USER_PREFERENCES.vibration === true;

  // Voorkeuren gelden als standaardwaarde; een geladen pad kan ze overschrijven.
  const PREFERRED_MODEL = USER_PREFERENCES.preferred_model
    ? normalizeModelName(USER_PREFERENCES.preferred_model)
    : DEFAULT_MODEL;
  const PREFERRED_MODEL_CONFIDENCE = (() => {
    const value = Number.parseFloat(USER_PREFERENCES.model_confidence);
    return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : DEFAULT_MODEL_CONFIDENCE;
  })();
  const PREFERRED_HEADING_FEEDBACK_FPS = (() => {
    const value = Number.parseFloat(USER_PREFERENCES.update_frequency);
    return Number.isFinite(value) ? Math.min(10, Math.max(0.2, value)) : DEFAULT_HEADING_FEEDBACK_FPS;
  })();
  currentModel = PREFERRED_MODEL;
  currentModelConfidence = PREFERRED_MODEL_CONFIDENCE;
  headingFeedbackFps = PREFERRED_HEADING_FEEDBACK_FPS;
  sendIntervalMs = Math.max(100, Math.round(1000 / headingFeedbackFps));
  // Zet de modelkeuzelijst op het voorkeursmodel (kan nog handmatig gewijzigd worden).
  if (modelSelectEl) {
    modelSelectEl.value = PREFERRED_MODEL;
  }
  let orientationWarningsEnabled = false;
  let noPathWarningsEnabled = true;
  let latencyWarningsEnabled = true;
  let sentFrames = 0;
  let framesSince = 0;
  let lastRateT = performance.now();
  const sentAtByFrameId = new Map();
  let lastLatency = null;
  let latencyAboveThresholdSinceMs = 0;
  let lastLatencyWarningAtMs = 0;
  let latestResultMasks = [];
  let trailPreviewRafId = null;
  let satelliteVisible = false;
  let lastArucoMarkerSpeechAtById = new Map();
  let lastArucoMarkerSpeechTextById = new Map();
  let lastArucoMarkerStateById = new Map();
  let arucoMarkerInstructionsById = new Map();
  let latestArucoMarkerDistance = 0;
  let arucoMarkerTypesById = new Map();
  let allowedArucoMarkerIds = new Set();
  let initialResumeApplied = false;
  let resumeAdvanceGuard = false;
  let lastArrivedMarkerId = null;
  let lastMarkerInstructionText = "";
  let noTrackerAlarmCounter = 0;

  function setStatus(message, tone = "") {
    statusEl.textContent = message;
    statusEl.className = `status${tone ? ` ${tone}` : ""}`;
  }

  function renderLatency() {
    latencyValueEl.textContent = Number.isFinite(lastLatency) ? `${lastLatency.toFixed(1)} ms` : "--";
  }

  function isPortraitOrientation() {
    if (window.screen?.orientation?.type) {
      return window.screen.orientation.type.startsWith("portrait");
    }
    return window.innerHeight >= window.innerWidth;
  }

  function renderOrientationStatus() {
    downAngleValueEl.textContent = latestPitchDeg === null ? "--" : `${latestPitchDeg.toFixed(1)} deg`;
    levelAngleValueEl.textContent = latestRollDeg === null ? "--" : `${latestRollDeg.toFixed(1)} deg`;
    portraitValueEl.textContent = isPortraitOrientation() ? "Yes" : "No";

    if (!orientationTrackingActive) {
      orientationStatusEl.textContent = "Kantelsensor niet gestart.";
      return;
    }
    if (latestPitchDeg === null || latestRollDeg === null) {
      orientationStatusEl.textContent = "Wachten op gegevens van de kantelsensor...";
      return;
    }

    const downErrorDeg = Math.abs(Math.abs(latestPitchDeg) - DOWN_ANGLE_TARGET_DEG);
    const levelErrorDeg = Math.abs(latestRollDeg);
    const portraitOk = isPortraitOrientation();
    const downOk = downErrorDeg <= DOWN_ANGLE_TOLERANCE_DEG;
    const levelOk = levelErrorDeg <= LEVEL_TOLERANCE_DEG;

    orientationStatusEl.textContent = `Kantelfout ${downErrorDeg.toFixed(1)} gr | Waterpas-fout ${levelErrorDeg.toFixed(1)} gr | Staand ${portraitOk ? "ok" : "niet ok"}${downOk && levelOk && portraitOk ? " | Klaar" : ""}`;
  }

  function maybeSpeakOrientationWarnings() {
    if (!orientationWarningsEnabled) {
      return;
    }
    if (!walkingActive || latestPitchDeg === null || latestRollDeg === null) {
      return;
    }

    const now = Date.now();
    const absPitchDeg = Math.abs(latestPitchDeg);
    if (absPitchDeg < 50 || absPitchDeg > 85) {
      if ((now - lastDownAngleWarningAtMs) >= 60000) {
        speak(
          currentLanguage === "en-GB"
            ? (absPitchDeg < 50
              ? "The camera points too much downward."
              : "The camera points too much upward. Tilt it downward.")
            : (absPitchDeg < 50
              ? "De camera wijst teveel naar beneden."
              : "De camera wijst teveel naar boven, kantel deze naar beneden."),
          "instruction"
        );
        lastDownAngleWarningAtMs = now;
      }
    }

    if (latestRollDeg < -50 || latestRollDeg > 50) {
      if ((now - lastLevelWarningAtMs) >= 60000) {
        speak(
          currentLanguage === "en-GB"
            ? "Camera is not straight. It should be placed vertically."
            : "Camera niet recht. Deze moet verticaal geplaatst worden.",
          "instruction"
        );
        lastLevelWarningAtMs = now;
      }
    }
  }

  function handleDeviceOrientation(event) {
    const beta = Number(event?.beta);
    const gamma = Number(event?.gamma);
    latestPitchDeg = Number.isFinite(beta) ? beta : null;
    latestRollDeg = Number.isFinite(gamma) ? gamma : null;
    renderOrientationStatus();
    maybeSpeakOrientationWarnings();
  }

  async function startOrientationTracking() {
    if (!("DeviceOrientationEvent" in window)) {
      orientationTrackingActive = false;
      orientationStatusEl.textContent = "Kantelsensor niet ondersteund.";
      return;
    }

    try {
      if (typeof DeviceOrientationEvent.requestPermission === "function") {
        const permission = await DeviceOrientationEvent.requestPermission();
        if (permission !== "granted") {
          orientationTrackingActive = false;
          orientationStatusEl.textContent = "Toegang tot kantelsensor geweigerd.";
          return;
        }
      }
    } catch {
      orientationTrackingActive = false;
      orientationStatusEl.textContent = "Toegang tot kantelsensor mislukt.";
      return;
    }

    if (!orientationListenerAttached) {
      window.addEventListener("deviceorientation", handleDeviceOrientation);
      orientationListenerAttached = true;
    }
    orientationTrackingActive = true;
    renderOrientationStatus();
  }

  function maybeSpeakLatencyWarning() {
    if (!latencyWarningsEnabled) {
      latencyAboveThresholdSinceMs = 0;
      return;
    }
    const now = Date.now();
    if (!Number.isFinite(lastLatency) || lastLatency <= LATENCY_WARNING_THRESHOLD_MS) {
      latencyAboveThresholdSinceMs = 0;
      return;
    }

    if (!latencyAboveThresholdSinceMs) {
      latencyAboveThresholdSinceMs = now;
      return;
    }

    if ((now - latencyAboveThresholdSinceMs) < 5000) {
      return;
    }

    if ((now - lastLatencyWarningAtMs) < 5000) {
      return;
    }

    lastLatencyWarningAtMs = now;
    const roundedLatency = Math.round(lastLatency);
    speak(
      currentLanguage === "en-GB"
        ? `Warning, latency ${roundedLatency} milliseconds.`
        : `Waarschuwing, vertraging ${roundedLatency} milliseconden.`,
      "instruction"
    );
  }

  function supportsHaptics() {
    return hapticsEnabled && typeof navigator !== "undefined" && typeof navigator.vibrate === "function";
  }

  function vibratePattern(pattern) {
    if (!supportsHaptics()) {
      return false;
    }
    return navigator.vibrate(pattern);
  }

  function cancelHaptics() {
    if (typeof navigator === "undefined" || typeof navigator.vibrate !== "function") {
      return;
    }
    navigator.vibrate(0);
  }

  function getHapticDirectionKeyForHeading(heading) {
    const normalizedHeading = toFiniteNumber(heading);
    if (normalizedHeading === null) {
      return null;
    }
    if (normalizedHeading >= referenceHeading + HAPTIC_COURSE_TOLERANCE_DEG) {
      return "left";
    }
    if (normalizedHeading <= referenceHeading - HAPTIC_COURSE_TOLERANCE_DEG) {
      return "right";
    }
    return "forward";
  }

  function triggerDirectionHaptic(heading, force = false) {
    const directionKey = getHapticDirectionKeyForHeading(heading);
    if (!directionKey) {
      lastHapticDirectionKey = null;
      lastHapticAtMs = 0;
      lastTurnHapticDirectionKey = null;
      lastTurnHapticCount = 0;
      return null;
    }

    const now = Date.now();
    let pattern = 20;
    let minIntervalMs = 2000;

    if (directionKey === "left") {
      pattern = 260;
      minIntervalMs = 450;
    } else if (directionKey === "right") {
      pattern = [80, 100, 80];
      minIntervalMs = 260;
    }

    const directionChanged = directionKey !== lastHapticDirectionKey;
    if (!force && !directionChanged && (now - lastHapticAtMs) < minIntervalMs) {
      return null;
    }

    lastHapticDirectionKey = directionKey;
    lastHapticAtMs = now;
    if (directionKey === "left" || directionKey === "right") {
      if (directionKey === lastTurnHapticDirectionKey) {
        lastTurnHapticCount += 1;
      } else {
        lastTurnHapticDirectionKey = directionKey;
        lastTurnHapticCount = 1;
      }
    } else {
      lastTurnHapticDirectionKey = null;
      lastTurnHapticCount = 0;
    }
    vibratePattern(pattern);
    return directionKey;
  }

  function triggerWarningHaptic() {
    vibratePattern([250, 120, 250, 120, 250]);
  }

  function updateWaypointNavButtons() {
    // Verberg "Vorig" op het eerste knooppunt en "Volgend" op het laatste; enkel zichtbaar tijdens het wandelen.
    const canPrev = walkingActive && activePointIndex > 0;
    const canNext = walkingActive && activePointIndex < routePoints.length - 1;
    prevWaypointBtn.classList.toggle("hidden", !canPrev);
    nextWaypointBtn.classList.toggle("hidden", !canNext);
  }

  function setWalkingChromeVisibility(isWalking) {
    startBtn.classList.toggle("hidden", isWalking);
    stopBtn.classList.toggle("hidden", !isWalking);
    updateWaypointNavButtons();
    if (topbarEl) {
      topbarEl.classList.toggle("hidden", isWalking);
    }
  }

  function updateBaseLayer() {
    if (satelliteVisible) {
      if (map.hasLayer(streetLayer)) {
        map.removeLayer(streetLayer);
      }
      if (!map.hasLayer(satelliteLayer)) {
        satelliteLayer.addTo(map);
      }
      satelliteToggleBtn.textContent = "Stratenkaart tonen";
    } else {
      if (map.hasLayer(satelliteLayer)) {
        map.removeLayer(satelliteLayer);
      }
      if (!map.hasLayer(streetLayer)) {
        streetLayer.addTo(map);
      }
      satelliteToggleBtn.textContent = "Satelliet tonen";
    }
  }

  function getLastSelectedPathSlug() {
    try {
      return window.localStorage.getItem(LAST_SELECTED_PATH_STORAGE_KEY) || "";
    } catch {
      return "";
    }
  }

  function getRequestedPathSlug() {
    try {
      const params = new URLSearchParams(window.location.search);
      return (params.get("slug") || "").trim();
    } catch {
      return "";
    }
  }

  function getQueryParam(name) {
    try {
      const params = new URLSearchParams(window.location.search);
      return (params.get(name) || "").trim();
    } catch {
      return "";
    }
  }

  function getResumeSessionId() {
    return getQueryParam("sessionID") || getQueryParam("sessionId") || getQueryParam("session_id");
  }

  function getResumePointIndex() {
    const value = Number.parseInt(getQueryParam("pointIndex") || "0", 10);
    return Number.isFinite(value) && value >= 0 ? value : 0;
  }

  function getResumePointId() {
    return getQueryParam("pointId");
  }

  function shouldResumeWalkingFromUrl() {
    return getQueryParam("walking") === "1" && Boolean(getResumeSessionId());
  }

  function applyResumePointFromUrl() {
    if (!routePoints.length) {
      activePointIndex = 0;
      return;
    }

    const pointId = getResumePointId();
    if (pointId) {
      const foundIndex = routePoints.findIndex((point) => String(point.id) === pointId);
      if (foundIndex >= 0) {
        activePointIndex = foundIndex;
        return;
      }
    }

    activePointIndex = Math.min(getResumePointIndex(), routePoints.length - 1);
  }

  function setLastSelectedPathSlug(slug) {
    try {
      if (slug) {
        window.localStorage.setItem(LAST_SELECTED_PATH_STORAGE_KEY, slug);
      } else {
        window.localStorage.removeItem(LAST_SELECTED_PATH_STORAGE_KEY);
      }
    } catch {}
  }

  function updateARSwitchLink() {
    if (!arSwitchLinkEl) return;
    const url = new URL("followpathAR.php", window.location.href);
    const slug = savedPathsEl.value || getRequestedPathSlug() || getLastSelectedPathSlug();

    if (slug) {
      url.searchParams.set("slug", slug);
    }

    if (walkingActive && currentSessionId) {
      url.searchParams.set("sessionID", currentSessionId);
      url.searchParams.set("pointIndex", String(activePointIndex));
      url.searchParams.set("walking", "1");

      const point = currentPoint();
      if (point?.id) {
        url.searchParams.set("pointId", String(point.id));
      }
    }

    arSwitchLinkEl.href = url.toString();
  }

  function normalizeHeading(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    return ((n % 360) + 360) % 360;
  }

  function normalizeModelName(value) {
    const normalized = String(value || "").trim().toLowerCase();
    if (normalized === "1") return "unrealsim";
    if (normalized === "2") return "laerbeekbos";
    if (normalized === "3") return "kaai";
    if (["unrealsim", "laerbeekbos", "kaai", "denham"].includes(normalized)) {
      return normalized;
    }
    return DEFAULT_MODEL;
  }

  function createSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return `sess-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }

  function isNonInterruptingSpeechKind(kind) {
    return kind === "direction" || kind === "marker";
  }

  function speak(text, kind = "general") {
    const content = String(text || "").trim();
    if (!content || !("speechSynthesis" in window)) return false;

    const nonInterrupting = isNonInterruptingSpeechKind(kind);
    if (nonInterrupting && activeSpeechKind && !isNonInterruptingSpeechKind(activeSpeechKind)) {
      return false;
    }
    if (kind === "instruction") {
      lastInstructionText = content;
    }

    const utterance = new SpeechSynthesisUtterance(content);
    const selectedLanguage = currentLanguage || DEFAULT_LANGUAGE;
    utterance.lang = selectedLanguage;
    utterance.rate = 1.0;

    const voices = window.speechSynthesis.getVoices();
    const matchingVoice = voices.find((voice) => voice.lang === selectedLanguage)
      || (selectedLanguage === "nl-BE"
        ? voices.find((voice) => voice.lang === "nl-BE" || voice.lang.startsWith("nl"))
        : voices.find((voice) => voice.lang === "en-GB" || voice.lang.startsWith("en")));

    if (matchingVoice) {
      utterance.voice = matchingVoice;
    }

    utterance.onstart = () => {
      activeSpeechKind = kind;
    };
    utterance.onend = () => {
      if (activeSpeechKind === kind) {
        activeSpeechKind = null;
      }
    };
    utterance.onerror = () => {
      if (activeSpeechKind === kind) {
        activeSpeechKind = null;
      }
    };

    window.speechSynthesis.speak(utterance);

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: "speak",
        sessionId: currentSessionId,
        text: content,
      }));
    }

    return true;
  }

  if ("speechSynthesis" in window) {
    window.speechSynthesis.onvoiceschanged = () => {};
  }

  function getStereoAudioContext() {
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) {
      return null;
    }
    if (!stereoAudioContext) {
      stereoAudioContext = new AudioContextCtor();
    }
    return stereoAudioContext;
  }

  async function ensureStereoAudioReady() {
    const context = getStereoAudioContext();
    if (!context) {
      return null;
    }
    if (context.state === "suspended") {
      try {
        await context.resume();
      } catch {}
    }
    return context;
  }

  function getStereoToneFrequency(directionKey, heading) {
    const normalizedHeading = toFiniteNumber(heading);
    const deviation = normalizedHeading === null
      ? 0
      : Math.min(90, Math.abs(normalizedHeading - referenceHeading));
    const ratio = deviation / 90;
    const baseFrequency = directionKey === "left" ? 300 : 600;   // links start op 300 Hz, rechts start op 600 Hz (C#5)
    const maxExtraFrequency = directionKey === "left" ? 100 : 150;
    return baseFrequency + (ratio * maxExtraFrequency);
  }

  function playStereoDirectionTone(directionKey, heading) {
    if (!beepsEnabled || directionKey !== "left" && directionKey !== "right") {
      return;
    }

    const context = getStereoAudioContext();
    if (!context || context.state !== "running") {
      return;
    }

    const now = context.currentTime;
    const oscillator = context.createOscillator();
    const gainNode = context.createGain();
    const panner = context.createStereoPanner();

    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(getStereoToneFrequency(directionKey, heading), now);
    panner.pan.setValueAtTime(directionKey === "left" ? -1 : 1, now);

    gainNode.gain.setValueAtTime(0.0001, now);
    gainNode.gain.exponentialRampToValueAtTime(0.18, now + 0.01);
    //gainNode.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);

    oscillator.connect(gainNode);
    gainNode.connect(panner);
    panner.connect(context.destination);

    oscillator.start(now);
    oscillator.stop(now + 0.2);
  }

  function playManualDirectionBeep(directionKey) {
    if (!beepsEnabled || directionKey !== "left" && directionKey !== "right") {
      return;
    }
    ensureStereoAudioReady().then(() => {
      playStereoDirectionTone(directionKey, directionKey === "left" ? 0 : 180);
    }).catch(() => {});
  }

  function playNoTrackingAlarm() {
    if (!beepsEnabled) {
      return;
    }
    const context = getStereoAudioContext();
    if (!context || context.state !== "running") {
      return;
    }

    const now = context.currentTime;
    const oscillator = context.createOscillator();
    const gainNode = context.createGain();
    const panner = context.createStereoPanner();
    const beepStarts = [0, 0.18, 0.36];
    const beepDuration = 0.08;

    oscillator.type = "square";
    oscillator.frequency.setValueAtTime(880, now);
    panner.pan.setValueAtTime(0, now);
    gainNode.gain.setValueAtTime(0.0001, now);

    beepStarts.forEach((offset) => {
      const beepStart = now + offset;
      const beepPeak = beepStart + 0.01;
      const beepEnd = beepStart + beepDuration;
      gainNode.gain.setValueAtTime(0.0001, beepStart);
      gainNode.gain.exponentialRampToValueAtTime(0.2, beepPeak);
      gainNode.gain.exponentialRampToValueAtTime(0.0001, beepEnd);
    });

    oscillator.connect(gainNode);
    gainNode.connect(panner);
    panner.connect(context.destination);

    oscillator.start(now);
    oscillator.stop(now + 0.5);
  }

  function drawArrowOnCanvas(canvasEl, canvasCtx, headingDeg) {
    const w = canvasEl.width;
    const h = canvasEl.height;
    const cx = w / 2;
    const cy = h / 2;

    canvasCtx.clearRect(0, 0, w, h);
    canvasCtx.beginPath();
    canvasCtx.arc(cx, cy, Math.max(24, Math.min(w, h) / 2 - 8), 0, Math.PI * 2);
    canvasCtx.strokeStyle = "rgba(255,255,255,0.35)";
    canvasCtx.lineWidth = 2;
    canvasCtx.stroke();

    canvasCtx.fillStyle = "rgba(255,255,255,0.7)";
    canvasCtx.font = "12px system-ui";
    canvasCtx.textAlign = "center";
    canvasCtx.fillText("Forward", cx, 14);

    if (typeof headingDeg !== "number" || Number.isNaN(headingDeg)) return;

    const angleRad = (-headingDeg * Math.PI) / 180;
    const arrowBody = Math.max(24, Math.min(w, h) * 0.28);
    const arrowTip = Math.max(36, Math.min(w, h) * 0.4);
    canvasCtx.save();
    canvasCtx.translate(cx, cy);
    canvasCtx.rotate(angleRad);
    canvasCtx.beginPath();
    canvasCtx.moveTo(-8, -3);
    canvasCtx.lineTo(arrowBody, -3);
    canvasCtx.lineTo(arrowBody, -10);
    canvasCtx.lineTo(arrowTip, 0);
    canvasCtx.lineTo(arrowBody, 10);
    canvasCtx.lineTo(arrowBody, 3);
    canvasCtx.lineTo(-8, 3);
    canvasCtx.closePath();
    canvasCtx.fillStyle = "#ff3b30";
    canvasCtx.fill();
    canvasCtx.restore();
  }

  function drawArrow(headingDeg) {
    drawArrowOnCanvas(mapCompass, mapCompCtx, headingDeg);
  }

  function syncTrailDirectionMode() {
    if (mapViewEnabled) {
      mapHeadingOverlayEl.classList.remove("hidden");
    }
    if (cameraViewEnabled) {
      mapCameraOverlayEl.classList.remove("hidden");
    }
  }

  function resizeTrailMaskOverlay() {
    const width = Math.max(1, Math.round(trailMaskOverlayEl.clientWidth));
    const height = Math.max(1, Math.round(trailMaskOverlayEl.clientHeight));
    if (trailMaskOverlayEl.width !== width || trailMaskOverlayEl.height !== height) {
      trailMaskOverlayEl.width = width;
      trailMaskOverlayEl.height = height;
    }
  }

  function clearTrailMaskOverlay() {
    resizeTrailMaskOverlay();
    trailMaskOverlayCtx.clearRect(0, 0, trailMaskOverlayEl.width, trailMaskOverlayEl.height);
  }

  function resizeTrailPreview() {
    const width = Math.max(1, Math.round(trailPreviewEl.clientWidth));
    const height = Math.max(1, Math.round(trailPreviewEl.clientHeight));
    if (trailPreviewEl.width !== width || trailPreviewEl.height !== height) {
      trailPreviewEl.width = width;
      trailPreviewEl.height = height;
    }
  }

  function clearTrailPreview() {
    resizeTrailPreview();
    trailPreviewCtx.fillStyle = "#000";
    trailPreviewCtx.fillRect(0, 0, trailPreviewEl.width, trailPreviewEl.height);
  }

  function drawTrailPreviewFrame() {
    resizeTrailPreview();
    if (trailVideoEl.videoWidth && trailVideoEl.videoHeight) {
      trailPreviewCtx.drawImage(trailVideoEl, 0, 0, trailPreviewEl.width, trailPreviewEl.height);
      return;
    }
    clearTrailPreview();
  }

  function scheduleTrailPreviewRender() {
    if (trailPreviewRafId !== null) {
      cancelAnimationFrame(trailPreviewRafId);
    }

    const tick = () => {
      drawTrailPreviewFrame();
      if (stream) {
        trailPreviewRafId = requestAnimationFrame(tick);
      } else {
        trailPreviewRafId = null;
      }
    };

    trailPreviewRafId = requestAnimationFrame(tick);
  }

  function stopTrailPreviewRender() {
    if (trailPreviewRafId !== null) {
      cancelAnimationFrame(trailPreviewRafId);
      trailPreviewRafId = null;
    }
    clearTrailPreview();
  }

  function isPointPair(value) {
    return Array.isArray(value)
      && value.length >= 2
      && Number.isFinite(Number(value[0]))
      && Number.isFinite(Number(value[1]));
  }

  function collectMaskPolygons(value, polygons = []) {
    if (!Array.isArray(value) || !value.length) {
      return polygons;
    }
    if (value.every((item) => isPointPair(item))) {
      polygons.push(value);
      return polygons;
    }
    value.forEach((item) => collectMaskPolygons(item, polygons));
    return polygons;
  }

  function drawTrailMaskOverlay(maskData) {
    resizeTrailMaskOverlay();
    trailMaskOverlayCtx.clearRect(0, 0, trailMaskOverlayEl.width, trailMaskOverlayEl.height);

    const polygons = collectMaskPolygons(maskData);
    if (!polygons.length) {
      return;
    }

    const canvasWidth = trailMaskOverlayEl.width;
    const canvasHeight = trailMaskOverlayEl.height;
    const scaleX = canvasWidth / TARGET_W;
    const scaleY = canvasHeight / TARGET_H;

    trailMaskOverlayCtx.fillStyle = "rgba(34, 211, 238, 0.28)";
    trailMaskOverlayCtx.strokeStyle = "rgba(34, 211, 238, 0.9)";
    trailMaskOverlayCtx.lineWidth = 2;

    polygons.forEach((polygon) => {
      let started = false;
      trailMaskOverlayCtx.beginPath();
      polygon.forEach((point) => {
        const x = Number(point[0]) * scaleX;
        const y = Number(point[1]) * scaleY;
        if (!started) {
          trailMaskOverlayCtx.moveTo(x, y);
          started = true;
        } else {
          trailMaskOverlayCtx.lineTo(x, y);
        }
      });
      if (started) {
        trailMaskOverlayCtx.closePath();
        trailMaskOverlayCtx.fill();
        trailMaskOverlayCtx.stroke();
      }
    });
  }

  function hasTrackedPath(maskData = latestResultMasks) {
    if (currentReturnMasks) {
      return collectMaskPolygons(maskData).length > 0;
    }

    const normalizedHeading = toFiniteNumber(latestHeading);
    if (normalizedHeading === null) {
      return false;
    }

    const headingAtOneDecimal = Number(normalizedHeading.toFixed(1));
    return headingAtOneDecimal !== 90.0 || consecutiveNoTrackingHeadingUpdates < 5;
  }

  function updateHeadingTrackingState() {
    if (currentReturnMasks) {
      consecutiveNoTrackingHeadingUpdates = 0;
      return;
    }

    const normalizedHeading = toFiniteNumber(latestHeading);
    if (normalizedHeading === null) {
      consecutiveNoTrackingHeadingUpdates = 0;
      return;
    }

    const headingAtOneDecimal = Number(normalizedHeading.toFixed(1));
    if (headingAtOneDecimal === 90.0) {
      consecutiveNoTrackingHeadingUpdates += 1;
      return;
    }

    consecutiveNoTrackingHeadingUpdates = 0;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function roundCoord(value) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(6) : "--";
  }

  function formatHeading(value) {
    return value === null || value === undefined || !Number.isFinite(Number(value))
      ? "--"
      : `${Number(value).toFixed(1)} deg`;
  }

  function formatDistanceForSpeech(distanceMeters) {
    const distance = Number(distanceMeters);
    if (!Number.isFinite(distance)) {
      return "";
    }
    if (distance >= 1000) {
      const kilometers = (distance / 1000).toFixed(1);
      return currentLanguage === "en-GB"
        ? `${kilometers} kilometers`
        : `${kilometers} kilometer`;
    }
    // For sub-meter distances, speak half a meter when appropriate
    const lang = (currentLanguage || DEFAULT_LANGUAGE || "nl-BE").toString().toLowerCase();
    const shortLang = lang.split("-")[0];
    if (distance < 1) {
      if (distance >= 0.5) {
        return shortLang === "en" ? "half a meter" : "een halve meter";
      }
      return shortLang === "en" ? "less than half a meter" : "minder dan een halve meter";
    }

    // Round to the nearest 0.5 meter to support 1.5, 2.5, etc.
    const roundedHalf = Math.round(distance * 2) / 2;
    const whole = Math.floor(roundedHalf);
    const frac = roundedHalf - whole;
    if (frac === 0) {
      if (shortLang === "en") {
        return whole === 1 ? "1 meter" : `${whole} meters`;
      }
      return `${whole} meter`;
    }

    // fractional 0.5
    if (shortLang === "en") {
      return `${whole}.5 meters`;
    }
    // Dutch uses comma for decimals
    return `${whole},5 meter`;
  }

  function formatDistanceForDisplay(distanceMeters) {
    const distance = Number(distanceMeters);
    if (!Number.isFinite(distance)) {
      return "--";
    }

    // Round to the nearest 0.5 meter to support 1.5, 2.5, etc.
    const roundedHalf = Math.round(distance * 2) / 2;
    const whole = Math.floor(roundedHalf);
    const frac = roundedHalf - whole;

    if (frac === 0) {
      return `${whole} m`;
    }

    // fractional 0.5 — use comma for Dutch, period for English
    const lang = (currentLanguage || DEFAULT_LANGUAGE || "nl-BE").toString().toLowerCase();
    const shortLang = lang.split("-")[0];
    if (shortLang === "en") {
      return `${whole}.5 m`;
    }
    // Dutch uses comma for decimals
    return `${whole},5 m`;
  }

  function toFiniteNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function normalizeArucoMarker(marker) {
    if (!marker || typeof marker !== "object") {
      return null;
    }

    const rawId = marker.id ?? marker.aruco_marker_id ?? marker.marker_id;
    const idText = String(rawId ?? "").trim();
    if (!idText) {
      return null;
    }

    const numericId = Number(idText);
    const id = Number.isFinite(numericId) ? String(Math.round(numericId)) : idText;
    const rawArea = marker.area_px2 ?? marker.aruco_marker_area_px2 ?? marker.area;
    const area = toFiniteNumber(rawArea);
    const rawDistance = marker.distance_m ?? marker.aruco_marker_distance_m ?? marker.distance;
    const distanceMeters = toFiniteNumber(rawDistance);

    const rawPos = marker.horizontal_position ?? marker.horizontalPosition ?? marker.aruco_marker_horizontal_position ?? marker.position;
    const pos = rawPos !== undefined && rawPos !== null ? String(rawPos).trim() : "";
    const horizontal_position = pos ? pos.toLowerCase() : null;

    return {
      id,
      area: area !== null && area >= 0 ? area : null,
      horizontal_position,
      distance_m: distanceMeters !== null && distanceMeters >= 0 ? distanceMeters : null
    };
  }

  function compareArucoMarkerIds(left, right) {
    const leftId = Number(left);
    const rightId = Number(right);
    if (Number.isFinite(leftId) && Number.isFinite(rightId)) {
      return leftId - rightId;
    }
    return String(left).localeCompare(String(right));
  }

  function collectArucoMarkers(payload) {
    const markers = [];
    const seenIds = new Set();
    const addMarker = (marker) => {
      const normalizedMarker = normalizeArucoMarker(marker);
      if (!normalizedMarker || seenIds.has(normalizedMarker.id) || !isArucoMarkerAllowed(normalizedMarker.id)) {
        return;
      }
      seenIds.add(normalizedMarker.id);
      markers.push(normalizedMarker);
    };

    if (Array.isArray(payload?.aruco_markers)) {
      payload.aruco_markers.forEach(addMarker);
    }

    if (payload?.aruco_marker_id !== undefined) {
      addMarker({
        id: payload.aruco_marker_id,
        area_px2: payload.aruco_marker_area_px2,
        horizontal_position: payload.aruco_marker_horizontal_position,
        distance_m: payload.aruco_marker_distance_m
      });
    }

    return markers;
  }

  function buildArucoMarkerSpeechText(marker, instruction = "") {
    const pos = (marker.horizontal_position || marker.horizontalPosition || "").toString().trim();
    const distanceText = Number.isFinite(marker.distance_m)
      ? formatDistanceForSpeech(marker.distance_m)
      : "";
    const lang = (currentLanguage || DEFAULT_LANGUAGE || "nl-BE").toString().toLowerCase();
    const shortLang = lang.split("-")[0];

    if (instruction) {
      const chosen = shortLang === "en"
        ? { nearby: "nearby.", at: (value) => `at ${value} hour` }
        : { nearby: "in de buurt.", at: (value) => `op ${value} uur` };
      const posPhrase = pos ? chosen.at(pos) : chosen.nearby;
      const distanceSentence = distanceText
        ? shortLang === "en"
          ? `and ${distanceText} away.`
          : `en ${distanceText} afstand.`
        : "";
      if (getArucoMarkerType(marker.id) === "direction") {

          $followTones = shortLang=== "en-GB" ? "Follow the tones to the marker." : "Volg de richtingstonen naar de marker.";
          return `Marker ${marker.id} ${posPhrase}${distanceSentence ? ` ${distanceSentence}, ${$followTones} ` :  ""}`.trim();
      } else {
         return `Marker ${marker.id} ${posPhrase}${distanceSentence ? ` ${distanceSentence}, ${instruction}` :  ""}`.trim();
      }
    }

    const baseText = pos
      ? shortLang === "en"
        ? `Aruco marker at ${pos}`
        : `Aruco-marker op ${pos}`
      : shortLang === "en"
        ? "Aruco marker nearby"
        : "Aruco-marker in de buurt";
    const distanceSentence = distanceText
      ? shortLang === "en"
        ? `, ${distanceText} away.`
        : `, ${distanceText} afstand.`
      : ".";
    return `${baseText}${distanceSentence}`.trim();
  }

  function normalizeArucoMarkerInstructionMap(markers) {
    const instructionMap = new Map();
    if (!Array.isArray(markers)) {
      return instructionMap;
    }

    markers.forEach((marker) => {
      const normalizedMarker = normalizeArucoMarker(marker);
      const instruction = String(marker?.instruction || "").trim();
      if (!normalizedMarker || !instruction) {
        return;
      }
      instructionMap.set(normalizedMarker.id, instruction);
    });

    return instructionMap;
  }

  function updateAllowedArucoMarkerIds(markers) {
    allowedArucoMarkerIds.clear();
    if (!Array.isArray(markers)) {
      return;
    }

    markers.forEach((marker) => {
      const normalizedMarker = normalizeArucoMarker(marker);
      if (!normalizedMarker || normalizedMarker.id === null || normalizedMarker.id === undefined) {
        return;
      }
      allowedArucoMarkerIds.add(String(normalizedMarker.id));
    });
  }

  function updateArucoMarkerTypes(markers) {
    arucoMarkerTypesById.clear();
    if (!Array.isArray(markers)) {
      return;
    }

    markers.forEach((marker) => {
      const normalizedMarker = normalizeArucoMarker(marker);
      if (!normalizedMarker || normalizedMarker.id === null || normalizedMarker.id === undefined) {
        return;
      }
      const rawType = String(marker?.type ?? marker?.markerType ?? "").trim().toLowerCase();
      arucoMarkerTypesById.set(String(normalizedMarker.id), rawType === "direction" ? "direction" : "instruction");
    });
  }

  function getArucoMarkerType(markerId) {
    if (markerId === null || markerId === undefined) {
      return null;
    }
    const normalizedId = String(markerId).trim();
    if (!normalizedId) {
      return null;
    }
    return arucoMarkerTypesById.get(normalizedId) || null;
  }

  function isArucoMarkerAllowed(markerId) {
    if (!allowedArucoMarkerIds.size) {
      return false;
    }
    if (markerId === null || markerId === undefined) {
      return false;
    }
    const normalizedId = String(markerId).trim();
    return normalizedId !== "" && allowedArucoMarkerIds.has(normalizedId);
  }

  function collectConfiguredArucoMarkerInstructions(markers) {
    return markers
      .map((marker) => ({
        id: marker.id,
        instruction: String(arucoMarkerInstructionsById.get(marker.id) || "").trim()
      }))
      .filter((marker) => marker.instruction);
  }

  function maybeSpeakArucoMarkers(payload, type = "instruction") {
    const markers = collectArucoMarkers(payload);
    if (!markers.length) {
      return;
    }

    const messages = [];
    const dueMarkerTexts = new Map();
    const now = Date.now();

    latestArucoMarkerDistance =  payload?.aruco_marker_distance_m ?? payload?.distance_m ?? latestArucoMarkerDistance;

    markers.forEach((marker) => {
      const markerInstruction = String(arucoMarkerInstructionsById.get(marker.id) || "").trim();
      lastMarkerInstructionText = markerInstruction;
      const text = buildArucoMarkerSpeechText(marker, markerInstruction);  
      

      if (!text) {
        return;
      }

      const lastText = lastArucoMarkerSpeechTextById.get(marker.id);
      const lastSpokenAt = lastArucoMarkerSpeechAtById.get(marker.id) || 0;
      const isDue = (now - lastSpokenAt) >= ARUCO_MARKER_SPEECH_COOLDOWN_MS;
      const shouldSpeak = isDue && (lastSpokenAt === 0 || text !== lastText || lastText === undefined);
      if (!shouldSpeak) {
        return;
      }

      messages.push(text);
      dueMarkerTexts.set(marker.id, text);
      lastArrivedMarkerId = null;
    });

    if (!messages.length) {
      return;
    }

    const instructionText = messages.join(" ");
    if (speak(instructionText, "marker")) {
      currentInstructionEl.textContent = instructionText;
      currentInstructionMetaEl.textContent = instructionText;
      //showHelperMessage(instructionText, "Aruco marker");
      const spokenAt = Date.now();
      dueMarkerTexts.forEach((text, id) => {
        if (text) {
          lastArucoMarkerSpeechTextById.set(id, text);
          lastArucoMarkerSpeechAtById.set(id, spokenAt);
        }
      });
    }
  }

  function getDistanceMeters(lat1, lon1, lat2, lon2) {
    const earthRadius = 6371000;
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return earthRadius * c;
  }

  function toRadians(value) {
    return value * Math.PI / 180;
  }

  function normalizeAngle(angle) {
    let normalized = angle;
    while (normalized > 180) normalized -= 360;
    while (normalized < -180) normalized += 360;
    return normalized;
  }

  function bearing(fromPoint, toPoint) {
    const lat1 = toRadians(fromPoint.lat);
    const lat2 = toRadians(toPoint.lat);
    const dLng = toRadians(toPoint.lng - fromPoint.lng);
    const y = Math.sin(dLng) * Math.cos(lat2);
    const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLng);
    const degrees = Math.atan2(y, x) * 180 / Math.PI;
    return (degrees + 360) % 360;
  }

  function buildWaypointInstruction(point, index) {
    const manualInstruction = String(point?.instruction || "").trim();
    if (!manualInstruction) {
      return "Geen instructie";
    } else {
      return manualInstruction;
    }
    return `${manualInstruction}`;
  }

  function distanceToPoint(point) {
    const lat = toFiniteNumber(latestLatitude);
    const lon = toFiniteNumber(latestLongitude);
    const pointLat = toFiniteNumber(point?.lat);
    const pointLon = toFiniteNumber(point?.lng ?? point?.lon);
    if (lat === null || lon === null || pointLat === null || pointLon === null) {
      return null;
    }
    return getDistanceMeters(lat, lon, pointLat, pointLon);
  }

  function distanceBetweenPoints(fromPoint, toPoint) {
    const fromLat = toFiniteNumber(fromPoint?.lat);
    const fromLon = toFiniteNumber(fromPoint?.lng ?? fromPoint?.lon);
    const toLat = toFiniteNumber(toPoint?.lat);
    const toLon = toFiniteNumber(toPoint?.lng ?? toPoint?.lon);
    if (fromLat === null || fromLon === null || toLat === null || toLon === null) {
      return null;
    }
    return getDistanceMeters(fromLat, fromLon, toLat, toLon);
  }

  function buildArrivalSpeech(point, index) {
    const selectedLanguage = currentLanguage || DEFAULT_LANGUAGE;
    const baseText = buildWaypointInstruction(point, index)
      || (selectedLanguage === "en-GB"
        ? `Reached point ${index + 1}.`
        : `Punt ${index + 1} bereikt.`);
    const nextPoint = routePoints[index + 1] || null;
    const nextDistance = nextPoint ? distanceBetweenPoints(point, nextPoint) : null;

    if (nextDistance === null) {
      return baseText;
    }

    const roundedDistance = Math.round(nextDistance);
    const nextDistanceText = selectedLanguage === "en-GB"
      ? `${roundedDistance} meters to the next waypoint.`
      : `${roundedDistance} meter tot het volgende instructiepunt.`;

    return `${baseText} ${nextDistanceText}`;
  }

  function buildPathSelectedSpeech(pathName) {
    const selectedLanguage = currentLanguage || DEFAULT_LANGUAGE;
    const safePathName = String(pathName || "").trim() || "route";
    return selectedLanguage === "en-GB"
      ? `Selected path: ${safePathName}.`
      : `Geselecteerd pad: ${safePathName}.`;
  }

  function renderSessionId() {
    sessionIdValueEl.textContent = currentSessionId || "--";
    copySessionBtn.disabled = !currentSessionId;
    renderHelpButton();
    updateARSwitchLink();
    sessionIdMetaEl.textContent = currentSessionId
      ? (helperEnabled
        ? "Er is een hulplink actief voor deze sessie."
        : "Druk op Vraag hulp om een hulplink te delen.")
      : "Start het wandelen om een live sessie-id aan te maken.";
  }

  function renderHelpButton() {
    // Vrij wandelen gebruikt geen hulp op afstand: knop en kaart blijven verborgen.
    helpBtn.classList.add("hidden");
    if (remoteAssistantCardEl) {
      remoteAssistantCardEl.classList.add("hidden");
    }
  }

  function buildRemoteHelpUrl() {
    const url = new URL("remoteHelp.php", window.location.href);
    url.searchParams.set("sessionID", currentSessionId);
    return url.toString();
  }

  async function copySessionId() {
    if (!currentSessionId) {
      setStatus("Geen actieve sessie-id om te kopiëren.", "warn");
      return;
    }

    try {
      await navigator.clipboard.writeText(currentSessionId);
      sessionIdMetaEl.textContent = "Sessie-id gekopieerd naar klembord.";
      setStatus("Sessie-id gekopieerd.", "ok");
    } catch (error) {
      console.warn("Clipboard copy failed:", error);
      sessionIdMetaEl.textContent = "Kopiëren van sessie-id mislukt.";
      setStatus("Kopiëren van sessie-id mislukt.", "warn");
    }
  }

  function shareHelpLink() {
    if (!currentSessionId) {
      setStatus("Geen actieve sessie-id om te delen.", "warn");
      return;
    }

    const helpUrl = buildRemoteHelpUrl();
    const message = `I need help on Stay On Trails. Open this remote assistant link: ${helpUrl}`;
    const whatsappUrl = `https://wa.me/?text=${encodeURIComponent(message)}`;
    helperEnabled = true;
    renderSessionId();
    window.open(whatsappUrl, "_blank", "noopener,noreferrer");
    sessionIdMetaEl.textContent = "Hulplink geopend in WhatsApp.";
    setStatus("Hulplink klaar om te delen.", "ok");
  }

  function disconnectHelper() {
    if (!currentSessionId) {
      setStatus("Geen actieve hulpsessie om te verbreken.", "warn");
      return;
    }

    currentSessionId = createSessionId();
    helperEnabled = false;
    showHelperMessage("");
    renderSessionId();
    setStatus("Hulp verbroken. Een nieuwe sessie-id is actief.", "ok");
  }

  function toggleHelper() {
    if (helperEnabled) {
      disconnectHelper();
      return;
    }
    shareHelpLink();
  }

  function renderHeadingSpeechToggle() {
    headingSpeechToggleBtn.textContent = headingSpeechEnabled ? "Spraak dempen" : "Spraak inschakelen";
  }


  function showHelperMessage(messageText, senderLabel = "") {
    const content = String(messageText || "").trim();
    if (!content) {
      helperMessageValueEl.textContent = "Geen bericht.";
      helperMessageMetaEl.textContent = "Berichten van de hulp op afstand verschijnen hier.";
      return;
    }
    lastHelperMessageText = content;
    helperMessageValueEl.textContent = content;
    currentInstructionEl.textContent = content;
    helperMessageMetaEl.textContent = senderLabel
      ? `Ontvangen van ${senderLabel}.`
      : "Ontvangen van de hulp op afstand.";
  }


  function handleManualHeadingCommand(payload) {
    const direction = String(payload?.direction || "").trim().toLowerCase();
    if (!["left", "right", "forward", "backward"].includes(direction)) {
      return false;
    }

    const spokenLabels = currentLanguage === "en-GB"
      ? { forward: "Go forward.", backward: "Go backward." }
      : { forward: "Ga vooruit.", backward: "Ga achteruit." };
    const displayLabels = currentLanguage === "en-GB"
      ? { left: "Left.", right: "Right.", forward: "Forward.", backward: "Backward." }
      : { left: "Links.", right: "Rechts.", forward: "Vooruit.", backward: "Achteruit." };

    showHelperMessage(displayLabels[direction], String(payload?.from || "remote assistant"));
    if (direction === "left" || direction === "right") {
      playManualDirectionBeep(direction);
      return true;
    }

    speak(spokenLabels[direction], "instruction");
    return true;
  }

  function handleSetWaypointCommand(payload) {
    const pointIndex = Number(payload?.pointIndex ?? payload?.waypointIndex ?? payload?.index);
    if (!Number.isInteger(pointIndex) || pointIndex < 0 || pointIndex >= routePoints.length) {
      showHelperMessage("Ongeldig waypoint gekozen door de hulp op afstand.", String(payload?.from || "hulp op afstand"));
      return false;
    }

    if (!activateWaypointFromClick(pointIndex)) {
      showHelperMessage(`De hulp op afstand koos punt ${pointIndex + 1}. Start het wandelen om dit waypoint te gebruiken.`, String(payload?.from || "hulp op afstand"));
      return false;
    }

    helperMessageValueEl.textContent = `Huidig waypoint ingesteld op punt ${pointIndex + 1}.`;
    helperMessageMetaEl.textContent = `Ontvangen van ${String(payload?.from || "hulp op afstand")}.`;
    return true;
  }

  function getDirectionLabelForHeading(heading) {
    const normalizedHeading = toFiniteNumber(heading);
    if (normalizedHeading === null) {
      return null;
    }

    const selectedLanguage = currentLanguage || DEFAULT_LANGUAGE;
    const labels = selectedLanguage === "en-GB"
      ? { left: "left", right: "right", straight: "straight" }
      : { left: "links", right: "rechts", straight: "rechtdoor" };

    if (normalizedHeading >= referenceHeading + COURSE_TOLERANCE_DEG) {
      return labels.left;
    }
    if (normalizedHeading <= referenceHeading - COURSE_TOLERANCE_DEG) {
      return labels.right;
    }
    return labels.straight;
  }

  // "Links wandelen" / "Midden wandelen" / "Rechts wandelen": zet de referentie-heading
  // op een vaste waarde (links 80°, midden 90°, rechts 100°) in plaats van stapsgewijs.
  const DEFAULT_REFERENCE_HEADING_DEG = 90;
  const WALK_LEFT_HEADING_DEG = 80;
  const WALK_CENTER_HEADING_DEG = 90;
  const WALK_RIGHT_HEADING_DEG = 100;

  function setReferenceHeading(targetHeadingDeg) {
    referenceHeading = targetHeadingDeg;

    let directionWord;
    if (targetHeadingDeg < DEFAULT_REFERENCE_HEADING_DEG) {
      directionWord = currentLanguage === "en-GB" ? "Left" : "Links";
    } else if (targetHeadingDeg > DEFAULT_REFERENCE_HEADING_DEG) {
      directionWord = currentLanguage === "en-GB" ? "Right" : "Rechts";
    } else {
      directionWord = currentLanguage === "en-GB" ? "Middle" : "Midden";
    }

    const spokenText = currentLanguage === "en-GB"
      ? `${directionWord} selected.`
      : `${directionWord} geselecteerd.`;
    setStatus(`${directionWord} geselecteerd (referentie ${referenceHeading}°)`, "ok");
    updateWalkButtonsPressed();
    announceForScreenReader(spokenText);
    speak(spokenText, "instruction");
    renderTrailDirection();
    updateDirectionSpeech(true);
  }

  // Zet aria-pressed op de actieve richtingsknop zodat TalkBack "geselecteerd" meldt.
  function updateWalkButtonsPressed() {
    [[walkLeftBtn, WALK_LEFT_HEADING_DEG], [walkCenterBtn, WALK_CENTER_HEADING_DEG], [walkRightBtn, WALK_RIGHT_HEADING_DEG]]
      .forEach(([btn, deg]) => {
        if (btn) btn.setAttribute("aria-pressed", String(referenceHeading === deg));
      });
  }

  // Laat een schermlezer (TalkBack) de tekst voorlezen via een aria-live regio.
  // Eerst leegmaken zodat dezelfde tekst opnieuw wordt aangekondigd.
  function announceForScreenReader(message) {
    if (!a11yAnnounceEl) return;
    a11yAnnounceEl.textContent = "";
    window.setTimeout(() => { a11yAnnounceEl.textContent = message; }, 50);
  }

  function renderTrailDirection() {
    const headingToDraw = latestMarkerHeading !== null ? latestMarkerHeading : latestHeading;
    const directionLabel = getDirectionLabelForHeading(headingToDraw);
    mapTrailDirectionValueEl.textContent = directionLabel || "--";
    drawArrow(headingToDraw);

    const isDirectionMarker = latestMarkerHeading !== null && getArucoMarkerType(latestMarkerId) === "direction";
    if (isDirectionMarker) {
      mapTrailDirectionMetaEl.textContent = latestMarkerId
        ? `Marker ${String(latestMarkerId)}, ${latestArucoMarkerDistance.toFixed(2)} m.`
        : "Marker";
      latestResultMasks = [];
      return;
    }

    drawTrailMaskOverlay(latestResultMasks);

    if (!walkingActive) {
      mapTrailDirectionMetaEl.textContent = "Segmentatiebegeleiding inactief.";
      return;
    }
    if (latestHeading === null) {
      mapTrailDirectionMetaEl.textContent = "Wachten op live segmentatierichting...";
      return;
    }
    if (!hasTrackedPath()) {
      mapTrailDirectionMetaEl.textContent = "Geen tracking gedetecteerd.";
      return;
    }
    mapTrailDirectionMetaEl.textContent = `Live heading ${formatHeading(latestHeading)}`;
  }

  function updateSendRate() {
    framesSince += 1;
    const now = performance.now();
    const dt = now - lastRateT;
    if (dt >= 1000) {
      const fps = framesSince / (dt / 1000);
      sendRateValueEl.textContent = `${fps.toFixed(1)} fps`;
      framesSince = 0;
      lastRateT = now;
    }
  }

  function updateDirectionSpeech(force = false) {
    const headingToUse = latestMarkerHeading !== null ? latestMarkerHeading : latestHeading;
    const directionLabel = getDirectionLabelForHeading(headingToUse);
    if (!directionLabel) {
      lastSpokenDirectionKey = null;
      lastHapticDirectionKey = null;
      lastHapticAtMs = 0;
      lastTurnHapticDirectionKey = null;
      lastTurnHapticCount = 0;
      renderTrailDirection();
      return;
    }

    const isMarkerDirection = latestMarkerHeading !== null;
    if (!isMarkerDirection && !hasTrackedPath()) {
      lastSpokenDirectionKey = null;
      lastHapticDirectionKey = null;
      lastHapticAtMs = 0;
      lastTurnHapticDirectionKey = null;
      lastTurnHapticCount = 0;
      cancelHaptics();
      renderTrailDirection();
      if (!noPathWarningsEnabled) {
        noTrackerAlarmCounter = 0;
        return;
      }

      noTrackerAlarmCounter += 1;

      if (noTrackerAlarmCounter >= 50 && noPathWarningsEnabled) {
        //playNoTrackingAlarm();
        speak(currentLanguage === "en-GB" ? "No path detected." : "Geen pad zichtbaar.", "warning");
        noTrackerAlarmCounter = 0;
      }
      return;
    }

    noTrackingWarningActive = false;

    const directionKey = `${directionLabel}|${currentLanguage || DEFAULT_LANGUAGE}`;
    renderTrailDirection();
    const hapticDirectionKey = triggerDirectionHaptic(headingToUse, force);
    if (hapticDirectionKey === "left" || hapticDirectionKey === "right") {
      playStereoDirectionTone(hapticDirectionKey, headingToUse);
    }
    if (!headingSpeechEnabled) {
      lastSpokenDirectionKey = null;
      return;
    }

    const isForward = directionLabel === "straight" || directionLabel === "rechtdoor";
    // Met geleidingstonen aan spreken we pas na de 3e beep in dezelfde richting.
    // Staan de tonen uit, dan zijn er geen beeps om op te wachten en spreken we
    // de links/rechts-instructie meteen bij de eerste detectie uit.
    const requiredTurnHapticCount = beepsEnabled ? 3 : 1;
    const shouldTriggerTurnCue = !isForward
      && hapticDirectionKey !== null
      && lastTurnHapticDirectionKey === hapticDirectionKey
      && lastTurnHapticCount === requiredTurnHapticCount;

    if (isForward) {
      lastSpokenDirectionKey = null;
      return;
    }

    if (!force && !shouldTriggerTurnCue) {
      return;
    }

    lastSpokenDirectionKey = directionKey;
    speak(directionLabel, "direction");
  }

  function currentPoint() {
    return routePoints[activePointIndex] || null;
  }

  function activateWaypointFromClick(index, spokenMessage) {
    if (!walkingActive || index < 0 || index >= routePoints.length) {
      return false;
    }

    activePointIndex = index;
    resumeAdvanceGuard = false;
    updateARSwitchLink();
    lastSpokenPointId = null;
    offRouteWarningActive = false;
    renderRoute();
    setStatus(`Verdergaan vanaf punt ${index + 1}.`, "ok");
    const fallbackMessage = currentLanguage === "en-GB" ? "Next waypoint selected." : "Volgend knooppunt geselecteerd.";
    speak(spokenMessage || fallbackMessage, "instruction");
    return true;
  }

  function goToPreviousWaypoint() {
    if (!walkingActive) {
      setStatus("Start eerst het wandelen.", "warn");
      return;
    }
    if (activePointIndex <= 0) {
      setStatus("Je bent al bij het eerste knooppunt.", "warn");
      return;
    }
    const target = activePointIndex - 1;
    activateWaypointFromClick(
      target,
      currentLanguage === "en-GB"
        ? `Previous waypoint selected, ${target + 1} of ${routePoints.length} waypoints.`
        : `Vorig knooppunt geselecteerd, ${target + 1} van ${routePoints.length} knooppunten.`
    );
  }

  function goToNextWaypoint() {
    if (!walkingActive) {
      setStatus("Start eerst het wandelen.", "warn");
      return;
    }
    if (activePointIndex >= routePoints.length - 1) {
      setStatus("Je bent al bij het laatste knooppunt.", "warn");
      return;
    }
    const target = activePointIndex + 1;
    activateWaypointFromClick(
      target,
      currentLanguage === "en-GB"
        ? `Next waypoint selected, ${target + 1} of ${routePoints.length} waypoints.`
        : `Volgend knooppunt geselecteerd, ${target + 1} van ${routePoints.length} knooppunten.`
    );
  }

  function renderPointList() {
    if (!routePoints.length) {
      return;
    }
  }

  function updateWaypointMarkers() {
    waypointMarkers.forEach((marker) => map.removeLayer(marker));
    waypointMarkers = [];

    routePoints.forEach((point, index) => {
      const isDone = index < activePointIndex;
      const isActive = index === activePointIndex;
      const marker = L.marker([point.lat, point.lng], {
        icon: L.divIcon({
          className: "",
          html: `<div class="waypointBubble${isActive ? " active" : isDone ? " done" : ""}">${index + 1}</div>`,
          iconSize: [28, 28],
          iconAnchor: [14, 14]
        }),
        title: point.instruction || `Point ${index + 1}`
      }).addTo(map);
      marker.on("click", () => {
        if (activateWaypointFromClick(index)) {
          return;
        }
        speak(buildWaypointInstruction(point, index), "instruction");
      });
      waypointMarkers.push(marker);
    });
  }

  function updatePathLine() {
    if (pathLine) {
      map.removeLayer(pathLine);
      pathLine = null;
    }
    if (routePoints.length >= 2) {
      pathLine = L.polyline(routePoints.map((point) => [point.lat, point.lng]), {
        color: "#22d3ee",
        weight: 4,
        opacity: 0.9
      }).addTo(map);
    }
  }

  function renderCurrentInstruction() {
    // Vrij wandelen heeft geen route/waypoints: toon de centreer-begeleiding en het opnametellertje.
    currentInstructionEl.textContent = "Vrij wandelen";
    currentInstructionMetaEl.textContent = walkingActive
      ? `Houd het pad gecentreerd. ${recordedTrack.length} GPS-punten opgenomen.`
      : "De begeleiding houdt je in het midden van het pad.";
    distanceValueEl.textContent = String(recordedTrack.length);
  }

  function renderRoute() {
    renderCurrentInstruction();
    renderPointList();
    updateWaypointMarkers();
    updatePathLine();
    renderTrailDirection();
    updateWaypointNavButtons();
  }

  function updateGpsUi() {
    if (latestLatitude === null || latestLongitude === null) {
      latValueEl.textContent = "--";
      lonValueEl.textContent = "--";
      accuracyValueEl.textContent = "--";
      gpsStatusEl.textContent = walkingActive ? "GPS: wachten op locatie..." : "GPS niet gestart.";
      renderCurrentInstruction();
      renderPointList();
      return;
    }

    latValueEl.textContent = latestLatitude.toFixed(6);
    lonValueEl.textContent = latestLongitude.toFixed(6);
    accuracyValueEl.textContent = latestAccuracy === null ? "--" : `${latestAccuracy.toFixed(1)} m`;
    gpsStatusEl.textContent = `GPS: ${latestLatitude.toFixed(6)}, ${latestLongitude.toFixed(6)}`;
    currentLocationMarker.setLatLng([latestLatitude, latestLongitude]);
    map.setView([latestLatitude, latestLongitude], 19);
    renderCurrentInstruction();
    renderPointList();
  }

  function maybeAdvanceRoute() {
    const point = currentPoint();
    if (!point || !walkingActive) {
      offRouteWarningActive = false;
      return;
    }

    const distance = distanceToPoint(point);
    if (distance === null) {
      offRouteWarningActive = false;
      return;
    }

    if (resumeAdvanceGuard) {
      if (distance <= arrivalRadiusMeters) {
        setStatus(`Hervat bij punt ${activePointIndex + 1}. Verwijder je van dit punt voordat automatisch verdergaan wordt ingeschakeld.`, "ok");
        renderCurrentInstruction();
        return;
      }
      resumeAdvanceGuard = false;
    }

    if (distance > OFF_ROUTE_WARNING_METERS) {
      if (!offRouteWarningActive) {
        setStatus("Je bent te ver van je route verwijderd", "warn");
        const distanceText = formatDistanceForSpeech(distance);
        triggerWarningHaptic();
        speak(
          currentLanguage === "en-GB"
            ? `You are too far away from your route. You are ${distanceText} from the next waypoint.`
            : `Je bent te ver van je route verwijderd. Je bent ${distanceText} van het volgende waypoint.`,
          "instruction"
        );
        offRouteWarningActive = true;
      }
    } else if (offRouteWarningActive) {
      setStatus(`Onderweg: ${currentPath?.name || "route"}.`, "ok");
      offRouteWarningActive = false;
    }

    if (distance <= arrivalRadiusMeters) {
      if (lastSpokenPointId !== point.id) {
        speak(buildArrivalSpeech(point, activePointIndex), "instruction");
        lastSpokenPointId = point.id;
      }

      activePointIndex += 1;
      resumeAdvanceGuard = false;
      updateARSwitchLink();
      renderRoute();

      if (activePointIndex >= routePoints.length) {
        setStatus("Route voltooid.", "ok");
        speak(currentLanguage === "en-GB" ? "You have reached the destination." : "Je hebt je bestemming bereikt.", "instruction");
        offRouteWarningActive = false;
      } else {
        setStatus(`Punt ${activePointIndex} bereikt. Volgende instructie is actief.`, "ok");
        offRouteWarningActive = false;
      }
    }
  }

  async function startCamera(preferredDeviceId = null) {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
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

    trailVideoEl.srcObject = stream;
    await trailVideoEl.play();
    scheduleTrailPreviewRender();

    const track = stream.getVideoTracks()[0];
    activeVideoDeviceId = track?.getSettings?.().deviceId ?? null;
  }

  function sendAuthMessage() {
    if (!ws || ws.readyState !== WebSocket.OPEN || authStarted) return;
    authStarted = true;
    ws.send(JSON.stringify({
      type: "auth",
      token: BEARER_TOKEN,
      "X-Room": SIGNALING_ROOM
    }));
  }

  function restartCaptureTimer() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    if (ws && ws.readyState === WebSocket.OPEN && isAuthenticated) {
      timer = setInterval(captureAndSend, sendIntervalMs);
    }
  }

  function beginAuthenticatedStreaming() {
    if (isAuthenticated) return;
    isAuthenticated = true;
    restartCaptureTimer();
    mapTrailDirectionMetaEl.textContent = "Live segmentatie verbonden.";
    streamMetaEl.textContent = `Ingesteld interval: ${(sendIntervalMs / 1000).toFixed(2)} s per frame | ${headingFeedbackFps.toFixed(1)} fps doel.`;
  }

  async function startSegmentationGuidance(options = {}) {
    if (timer || (ws && ws.readyState === WebSocket.OPEN)) {
      return;
    }

    const reuseExistingSession = options.reuseExistingSession === true && Boolean(currentSessionId);

    trailCapEl.width = TARGET_W;
    trailCapEl.height = TARGET_H;
    currentSessionId = reuseExistingSession ? currentSessionId : createSessionId();
    helperEnabled = false;
    updateARSwitchLink();
    renderSessionId();
    isAuthenticated = false;
    authStarted = false;
    latestHeading = null;
    latestMarkerHeading = null;
    latestMarkerId = null;
    lastArucoMarkerSpeechAtById.clear();
    lastArucoMarkerSpeechTextById.clear();
    lastArucoMarkerStateById.clear();
    consecutiveNoTrackingHeadingUpdates = 0;
    lastSpokenDirectionKey = null;
    lastHapticDirectionKey = null;
    lastHapticAtMs = 0;
    lastTurnHapticDirectionKey = null;
    lastTurnHapticCount = 0;
    lastLatency = null;
    latencyAboveThresholdSinceMs = 0;
    lastLatencyWarningAtMs = 0;
    renderLatency();
    latestResultMasks = [];
    sentFrames = 0;
    framesSince = 0;
    lastRateT = performance.now();
    sentFramesValueEl.textContent = "0";
    sendRateValueEl.textContent = "0.0 fps";
    streamMetaEl.textContent = `Ingesteld interval: ${(sendIntervalMs / 1000).toFixed(2)} s per frame | ${headingFeedbackFps.toFixed(1)} fps doel.`;
    renderTrailDirection();

    try {
      await startCamera(activeVideoDeviceId);
    } catch (error) {
      console.error("Camera error:", error);
      mapTrailDirectionMetaEl.textContent = "Camera niet beschikbaar voor segmentatiebegeleiding.";
      return;
    }

    try {
      ws = new WebSocket(SIGNALING_SERVER);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        sendAuthMessage();
      };

      ws.onerror = (error) => {
        console.error("WS error", error);
        mapTrailDirectionMetaEl.textContent = "Segmentatie-websocketfout.";
      };

      ws.onclose = () => {
        if (walkingActive) {
          mapTrailDirectionMetaEl.textContent = "Segmentatiebegeleiding verbroken.";
        }
        stopSegmentationGuidance(false);
        renderTrailDirection();
      };

      ws.onmessage = (msg) => {
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
            payload?.authenticated === true ||
            payload?.type === "room_joined"
          ) {
            beginAuthenticatedStreaming();
            return;
          }

          if (
            payload?.type === "auth_error" ||
            payload?.type === "unauthorized" ||
            payload?.authenticated === false
          ) {
            mapTrailDirectionMetaEl.textContent = "Segmentatie-authenticatie mislukt.";
            stopSegmentationGuidance(true);
            return;
          }

          if (payload?.type === "helper_message") {
            const incomingSessionId = payload?.sessionId ?? payload?.session_id ?? null;
            if (currentSessionId && incomingSessionId && incomingSessionId !== currentSessionId) {
              return;
            }
            if (payload?.commandType === "manual_heading") {
              handleManualHeadingCommand(payload);
              return;
            }
            if (payload?.commandType === "set_waypoint") {
              handleSetWaypointCommand(payload);
              return;
            }
            const helperText = String(payload?.message || "").trim();
            if (!helperText) {
              return;
            }
            showHelperMessage(helperText, String(payload?.from || "remote assistant"));
            speak(helperText, "instruction");
            return;
          }

          const incomingSessionId = payload?.sessionId ?? payload?.session_id ?? null;
          if (currentSessionId && incomingSessionId && incomingSessionId !== currentSessionId) {
            return;
          }
          if (
            currentSessionId &&
            !incomingSessionId &&
            (
              payload?.heading !== undefined ||
              payload?.frame_id !== undefined ||
              payload?.aruco_marker_id !== undefined ||
              payload?.aruco_markers !== undefined
            )
          ) {
            return;
          }


          maybeSpeakArucoMarkers(payload, "instruction");


          // only update marker heading if it's present in the payload
          if (payload?.marker_heading !== undefined) {
            const markerId = payload?.aruco_marker_id ?? payload?.marker_id ?? payload?.id ?? payload?.markerId ?? null;
            if (!isArucoMarkerAllowed(markerId)) {
              latestMarkerHeading = null;
              latestMarkerId = null;
            } else {
               if (latestArucoMarkerDistance.toFixed(2) <= ARUCO_DISTANCE_SPEECH_THRESHOLD_METERS) {
                    if (lastArrivedMarkerId !== markerId) {
                      Instruction = "OK," + lastMarkerInstructionText;
                      speak(Instruction, "marker");
                      lastArrivedMarkerId = markerId;
                  }                  
                }              
              // Only update heading if type is "direction"
              if (getArucoMarkerType(markerId) === "direction") {
                const normalizedMarkerHeading = normalizeHeading(payload?.marker_heading);
                latestMarkerHeading = normalizedMarkerHeading;
                latestMarkerId = markerId;
                
              } else {
                latestMarkerHeading = null;
                latestMarkerId = null;
              }
            }
            updateDirectionSpeech();
          } else {
            latestMarkerHeading = null;
            latestMarkerId = null;
            if (payload?.resultMasks !== undefined || payload?.returnMasks !== undefined) {
              latestResultMasks = payload?.resultMasks ?? payload?.returnMasks ?? [];
              renderTrailDirection();
            } else if (payload?.frame_id !== undefined) {
              latestResultMasks = [];
              renderTrailDirection();
            }
          }

          const normalized = normalizeHeading(payload?.heading);
          if (normalized !== null) {
            latestHeading = normalized;
            updateHeadingTrackingState();
            updateDirectionSpeech();
          }


          const frameId = payload?.frame_id;
          if (frameId !== null && frameId !== undefined) {
            const sentAt = sentAtByFrameId.get(String(frameId));
            if (typeof sentAt === "number") {
              lastLatency = Math.max(0, performance.now() - sentAt);
              renderLatency();
              maybeSpeakLatencyWarning();
            }
            sentAtByFrameId.delete(String(frameId));
          }
        } catch {}
      };
    } catch (error) {
      console.error("WS connect failed", error);
      mapTrailDirectionMetaEl.textContent = "Kan segmentatiebegeleiding niet verbinden.";
      stopSegmentationGuidance(true);
    }
  }

  function stopSegmentationGuidance(resetUi = true) {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    if (ws) {
      const socket = ws;
      ws = null;
      try {
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
          socket.close();
        }
      } catch {}
    }
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    stopTrailPreviewRender();

    sentAtByFrameId.clear();
    nextFrameId = 1;
    currentSessionId = null;
    helperEnabled = false;
    renderSessionId();
    isAuthenticated = false;
    authStarted = false;
    latestHeading = null;
    latestMarkerHeading = null;
    latestMarkerId = null;
    lastArucoMarkerSpeechAtById.clear();
    lastArucoMarkerSpeechTextById.clear();
    lastArucoMarkerStateById.clear();
    consecutiveNoTrackingHeadingUpdates = 0;
    lastSpokenDirectionKey = null;
    lastHapticDirectionKey = null;
    lastHapticAtMs = 0;
    lastTurnHapticDirectionKey = null;
    lastTurnHapticCount = 0;
    lastLatency = null;
    latencyAboveThresholdSinceMs = 0;
    lastLatencyWarningAtMs = 0;
    renderLatency();
    latestResultMasks = [];
    framesSince = 0;
    sendRateValueEl.textContent = "0.0 fps";
    streamMetaEl.textContent = `Ingesteld interval: ${(sendIntervalMs / 1000).toFixed(2)} s per frame | ${headingFeedbackFps.toFixed(1)} fps doel.`;
    if (resetUi) {
      renderTrailDirection();
    }
  }

  function captureAndSend() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !isAuthenticated) return;
    if (!trailVideoEl.videoWidth || !trailVideoEl.videoHeight) return;

    trailCapCtx.drawImage(trailVideoEl, 0, 0, TARGET_W, TARGET_H);

    trailCapEl.toBlob(async (blob) => {
      if (!blob) return;

      try {
        const frameId = String(nextFrameId++);
        const buf = await blob.arrayBuffer();
        sentAtByFrameId.set(frameId, performance.now());

        ws.send(JSON.stringify({
          type: "frame_meta",
          frame_id: frameId,
          sessionId: currentSessionId,
          latitude: latestLatitude,
          longitude: latestLongitude,
          gps_accuracy: latestAccuracy,
          model: currentModel,
          confidence: currentModelConfidence,
          source: "live_camera",
          lastlatency: lastLatency,
          returnMasks: currentReturnMasks,
          sendMQTT: currentSendMQTT
        }));

        ws.send(buf);
        sentFrames += 1;
        sentFramesValueEl.textContent = String(sentFrames);
        updateSendRate();
      } catch (error) {
        console.error("Failed to send segmentation frame:", error);
      }
    }, "image/jpeg", JPEG_QUALITY);
  }

  function startLocationTracking() {
    if (!("geolocation" in navigator) || geoWatchId !== null) {
      return;
    }

    try {
      geoWatchId = navigator.geolocation.watchPosition(
        (position) => {
          const lat = Number(position?.coords?.latitude);
          const lon = Number(position?.coords?.longitude);
          const accuracy = Number(position?.coords?.accuracy);
          latestLatitude = Number.isFinite(lat) ? lat : null;
          latestLongitude = Number.isFinite(lon) ? lon : null;
          latestAccuracy = Number.isFinite(accuracy) ? accuracy : null;
          if (walkingActive && latestLatitude !== null && latestLongitude !== null) {
            const elevation = Number(position?.coords?.altitude);
            recordedTrack.push({
              lat: latestLatitude,
              lon: latestLongitude,
              ele: Number.isFinite(elevation) ? elevation : null,
              time: new Date().toISOString()
            });
            updateSaveGpxButton();
          }
          updateGpsUi();
          maybeAdvanceRoute();
        },
        (error) => {
          console.warn("Geolocation unavailable:", error);
          gpsStatusEl.textContent = "GPS: toegang geweigerd of niet beschikbaar";
          setStatus("GPS-toegang geweigerd of niet beschikbaar.", "warn");
        },
        { enableHighAccuracy: true, maximumAge: 3000, timeout: 10000 }
      );
    } catch (error) {
      console.warn("Failed to start geolocation:", error);
      gpsStatusEl.textContent = "GPS: starten mislukt";
      setStatus("Starten van GPS-tracking mislukt.", "warn");
    }
  }

  function stopLocationTracking() {
    if (geoWatchId !== null) {
      try { navigator.geolocation.clearWatch(geoWatchId); } catch {}
      geoWatchId = null;
    }
    walkingActive = false;
    latestLatitude = null;
    latestLongitude = null;
    latestAccuracy = null;
    updateGpsUi();
  }

  async function loadSavedPaths() {
    const response = await fetch(`${API_URL}?action=list_paths`);
    const payload = await response.json();
    if (!payload?.ok) {
      setStatus(payload?.error || "Ophalen van opgeslagen routes mislukt.", "warn");
      return;
    }

    savedPathsEl.innerHTML = `<option value="">Choose a saved path...</option>` + payload.paths.map((item) => `
      <option value="${escapeHtml(item.slug)}">${escapeHtml(item.name)} (${item.pointCount})</option>
    `).join("");

    const requestedSlug = getRequestedPathSlug();
    const preferredSlug = requestedSlug || getLastSelectedPathSlug();
    if (preferredSlug && payload.paths.some((item) => item.slug === preferredSlug)) {
      savedPathsEl.value = preferredSlug;
      await loadSelectedPath();

      if (shouldResumeWalkingFromUrl() && routePoints.length && currentSessionId) {
        startWalking({ resume: true }).catch(() => {
          setStatus("Kan wandelsessie niet hervatten.", "warn");
        });
      }
    } else if (requestedSlug) {
      setStatus("De gevraagde route is niet gevonden.", "warn");
    }
  }

  async function loadSelectedPath() {
    const slug = savedPathsEl.value;
    if (!slug) {
      setStatus("Kies eerst een route.", "warn");
      updateARSwitchLink();
      return;
    }

    const response = await fetch(`${API_URL}?action=load_path&slug=${encodeURIComponent(slug)}`);
    const payload = await response.json();
    if (!payload?.ok || !payload.path) {
      setStatus(payload?.error || "Laden van route mislukt.", "warn");
      return;
    }
    setLastSelectedPathSlug(slug);
    updateARSwitchLink();

    currentPath = payload.path;
    currentLanguage = payload.path.language === "en-GB" ? "en-GB" : DEFAULT_LANGUAGE;
    arucoMarkerInstructionsById = normalizeArucoMarkerInstructionMap(payload.path.arucoMarkers);
    updateAllowedArucoMarkerIds(payload.path.arucoMarkers);
    updateArucoMarkerTypes(payload.path.arucoMarkers);
    lastArucoMarkerSpeechAtById.clear();
    lastArucoMarkerSpeechTextById.clear();
    lastArucoMarkerStateById.clear();
    const loadedModelName = String(payload.path.model || "").trim();
    currentModel = loadedModelName ? normalizeModelName(loadedModelName) : PREFERRED_MODEL;
    const loadedModelConfidence = Number.parseFloat(payload.path.modelConfidence);
    currentModelConfidence = Number.isFinite(loadedModelConfidence)
      ? Math.min(1, Math.max(0, loadedModelConfidence))
      : PREFERRED_MODEL_CONFIDENCE;
    currentReturnMasks = payload.path.returnMasks === true;
    currentSendMQTT = payload.path.sendMQTT === true;
    syncTrailDirectionMode();
    const loadedHeadingFeedbackFps = Number.parseFloat(payload.path.headingFeedbackFps);
    headingFeedbackFps = Number.isFinite(loadedHeadingFeedbackFps)
      ? Math.min(10, Math.max(0.2, loadedHeadingFeedbackFps))
      : PREFERRED_HEADING_FEEDBACK_FPS;
    const loadedArrivalRadiusMeters = Number.parseInt(payload.path.arrivalRadiusMeters, 10);
    arrivalRadiusMeters = ALLOWED_ARRIVAL_RADIUS_METERS.includes(loadedArrivalRadiusMeters)
      ? loadedArrivalRadiusMeters
      : DEFAULT_ARRIVAL_RADIUS_METERS;
    sendIntervalMs = Math.max(100, Math.round(1000 / headingFeedbackFps));
    routePoints = Array.isArray(payload.path.points) ? payload.path.points.map((point, index) => ({
      id: point.id || `point-${index + 1}`,
      lat: Number(point.lat),
      lng: Number(point.lng ?? point.lon),
      instruction: String(point.instruction || "").trim()
    })).filter((point) => Number.isFinite(point.lat) && Number.isFinite(point.lng)) : [];

    const isResumeRequest = shouldResumeWalkingFromUrl();
    if (isResumeRequest && !initialResumeApplied) {
      currentSessionId = getResumeSessionId();
      applyResumePointFromUrl();
      resumeAdvanceGuard = true;
      initialResumeApplied = true;
    } else {
      activePointIndex = 0;
      resumeAdvanceGuard = false;
    }

    lastSpokenPointId = null;
    showHelperMessage("");
    renderSessionId();
    renderRoute();

    if (routePoints.length) {
      map.fitBounds(routePoints.map((point) => [point.lat, point.lng]), { padding: [40, 40] });
      setStatus(`${payload.path.name || slug} geladen.`, "ok");
      if (!isResumeRequest) {
        speak(buildPathSelectedSpeech(payload.path.name || slug), "instruction");
      }
    } else {
      setStatus("Deze route bevat geen geldige punten.", "warn");
    }
  }

  function pad2(value) {
    return String(value).padStart(2, "0");
  }

  function updateSaveGpxButton() {
    distanceValueEl.textContent = String(recordedTrack.length);
    saveGpxBtn.classList.toggle("hidden", recordedTrack.length === 0);
  }

  function buildGpxDocument() {
    const created = new Date().toISOString();
    // De XML-declaratie wordt in stukken opgebouwd zodat de PHP-parser ze niet als open-tag ziet.
    const header =
      "<" + "?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" +
      "<gpx version=\"1.1\" creator=\"Stay On Trails - Vrij wandelen\" xmlns=\"http://www.topografix.com/GPX/1/1\">\n" +
      "  <metadata><time>" + created + "</time></metadata>\n" +
      "  <trk>\n    <name>Vrije wandeling " + created + "</name>\n    <trkseg>\n";
    const body = recordedTrack.map((p) => {
      const ele = (p.ele !== null && p.ele !== undefined) ? "<ele>" + p.ele.toFixed(1) + "</ele>" : "";
      const time = p.time ? "<time>" + p.time + "</time>" : "";
      return "      <trkpt lat=\"" + p.lat.toFixed(7) + "\" lon=\"" + p.lon.toFixed(7) + "\">" + ele + time + "</trkpt>\n";
    }).join("");
    const footer = "    </trkseg>\n  </trk>\n</gpx>\n";
    return header + body + footer;
  }

  function saveGpx() {
    if (!recordedTrack.length) {
      setStatus("Er zijn nog geen GPS-punten opgenomen.", "warn");
      return;
    }
    const gpx = buildGpxDocument();
    const blob = new Blob([gpx], { type: "application/gpx+xml" });
    const url = URL.createObjectURL(blob);
    const now = new Date();
    const stamp = now.getFullYear() + pad2(now.getMonth() + 1) + pad2(now.getDate()) +
      "-" + pad2(now.getHours()) + pad2(now.getMinutes()) + pad2(now.getSeconds());
    const link = document.createElement("a");
    link.href = url;
    link.download = "vrije-wandeling-" + stamp + ".gpx";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 2000);
    setStatus(`GPX opgeslagen met ${recordedTrack.length} punten.`, "ok");
  }

  async function startWalking() {
    currentModel = normalizeModelName(modelSelectEl.value);
    recordedTrack = [];
    updateSaveGpxButton();

    ensureStereoAudioReady().catch(() => {});
    startOrientationTracking().catch(() => {});
    lastDownAngleWarningAtMs = 0;
    lastLevelWarningAtMs = 0;

    activePointIndex = 0;
    resumeAdvanceGuard = false;
    currentSessionId = null;

    walkingActive = true;
    updateARSwitchLink();
    lastSpokenPointId = null;
    offRouteWarningActive = false;
    setWalkingChromeVisibility(true);
    modelFieldEl.classList.add("hidden");
    setStatus("Onderweg. Houd het pad gecentreerd.", "ok");
    speak(currentLanguage === "en-GB" ? "Free walk started." : "Vrij wandelen gestart.", "instruction");

    startLocationTracking();
    await startSegmentationGuidance({});
    renderRoute();
  }

  function stopWalking() {
    stopLocationTracking();
    stopSegmentationGuidance();
    orientationTrackingActive = false;
    lastDownAngleWarningAtMs = 0;
    lastLevelWarningAtMs = 0;
    renderOrientationStatus();
    setWalkingChromeVisibility(false);
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    cancelHaptics();
    activeSpeechKind = null;
    activePointIndex = 0;
    resumeAdvanceGuard = false;
    updateARSwitchLink();
    lastSpokenPointId = null;
    offRouteWarningActive = false;
    lastHapticDirectionKey = null;
    lastHapticAtMs = 0;
    lastTurnHapticDirectionKey = null;
    lastTurnHapticCount = 0;
    consecutiveNoTrackingHeadingUpdates = 0;
    latencyAboveThresholdSinceMs = 0;
    lastLatencyWarningAtMs = 0;
    referenceHeading = DEFAULT_REFERENCE_HEADING_DEG;
    showHelperMessage("");
    renderRoute();
    renderHelpButton();
    // Modelkeuze blijft verborgen: het model wordt via de voorkeuren bepaald.
    updateSaveGpxButton();
    setStatus(
      recordedTrack.length
        ? `Wandeling gestopt. ${recordedTrack.length} GPS-punten opgenomen — sla de GPX op.`
        : "Wandeling gestopt.",
      "ok"
    );
  }

  function repeatLastInstruction() {
    if (!lastInstructionText) {
      setStatus("Er is nog geen instructie uitgesproken.", "warn");
      return;
    }
    speak(lastInstructionText, "instruction");
    setStatus("Laatste instructie herhaald.", "ok");
  }

  function toggleHeadingSpeech() {
    headingSpeechEnabled = !headingSpeechEnabled;
    if (!headingSpeechEnabled && "speechSynthesis" in window && activeSpeechKind === "direction") {
      window.speechSynthesis.cancel();
      activeSpeechKind = null;
    }
    renderHeadingSpeechToggle();
    setStatus(
      headingSpeechEnabled ? "Spraak ingeschakeld." : "Spraak gedempt.",
      "ok"
    );
  }

  startBtn.addEventListener("click", () => {
    startWalking().catch(() => setStatus("Kan het wandelen niet starten.", "warn"));
  });
  helpBtn.addEventListener("click", toggleHelper);
  repeatBtn.addEventListener("click", repeatLastInstruction);
  headingSpeechToggleBtn.addEventListener("click", toggleHeadingSpeech);
  walkLeftBtn.addEventListener("click", () => setReferenceHeading(WALK_LEFT_HEADING_DEG));
  walkCenterBtn.addEventListener("click", () => setReferenceHeading(WALK_CENTER_HEADING_DEG));
  walkRightBtn.addEventListener("click", () => setReferenceHeading(WALK_RIGHT_HEADING_DEG));
  copySessionBtn.addEventListener("click", () => {
    copySessionId().catch(() => {
      sessionIdMetaEl.textContent = "Kopiëren van sessie-id mislukt.";
      setStatus("Kopiëren van sessie-id mislukt.", "warn");
    });
  });
  satelliteToggleBtn.addEventListener("click", () => {
    satelliteVisible = !satelliteVisible;
    updateBaseLayer();
  });
  stopBtn.addEventListener("click", stopWalking);
  saveGpxBtn.addEventListener("click", saveGpx);
  prevWaypointBtn.addEventListener("click", goToPreviousWaypoint);
  nextWaypointBtn.addEventListener("click", goToNextWaypoint);
  savedPathsEl.addEventListener("change", () => {
    updateARSwitchLink();
    loadSelectedPath().catch(() => setStatus("Laden van route mislukt.", "warn"));
  });

  renderRoute();
  renderSessionId();
  renderHeadingSpeechToggle();
  renderLatency();
  renderOrientationStatus();
  if (!mapViewEnabled && mapWrapEl) {
    mapWrapEl.classList.add("hidden");
    // "Satelliet tonen" heeft enkel zin wanneer de kaart zichtbaar is.
    satelliteToggleBtn.classList.add("hidden");
  }
  if (!cameraViewEnabled) {
    mapCameraOverlayEl.classList.add("hidden");
  }
  syncTrailDirectionMode();
  setWalkingChromeVisibility(false);
  drawArrow(null);
  clearTrailPreview();
  clearTrailMaskOverlay();
  updateBaseLayer();
  updateARSwitchLink();
  window.addEventListener("resize", () => {
    drawTrailPreviewFrame();
    renderTrailDirection();
    renderOrientationStatus();
  });
  window.addEventListener("orientationchange", renderOrientationStatus);
  // Vrij wandelen: geen route laden. Model wordt gekozen via de keuzelijst.
  currentModel = normalizeModelName(modelSelectEl.value);
  updateSaveGpxButton();
</script>
</body>
</html>
