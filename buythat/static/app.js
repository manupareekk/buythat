const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), still = $("still"), viewer = document.querySelector(".viewer");
let cfg = {}, last = null, selected = null, recorder = null, chunks = [], pendingFile = null, ppLoaded = false;
const COLORS = { screen: "#6aa8ff", speech: "#c38bff", grounding: "#ffd34d", retrieval: "#5fd18b" };

async function init() {
  cfg = await (await fetch("/api/config")).json();
  $("meta").textContent = `catalog:${cfg.catalog} · paypal:${cfg.paypal_client_id ? cfg.paypal_env : "off"}`;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1440 } }, audio: true });
    video.srcObject = stream;
    const audio = new MediaStream(stream.getAudioTracks());
    recorder = new MediaRecorder(audio);
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  } catch (e) {
    $("hint").textContent = "Camera/mic unavailable here (needs https). Use a photo + typed command.";
  }
}

function grab() {
  if (pendingFile) return Promise.resolve(pendingFile);
  const c = document.createElement("canvas");
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
}

const talk = $("talk");
let frameBlob = null;
talk.addEventListener("pointerdown", async (e) => {
  e.preventDefault();
  if (!recorder || talk.classList.contains("busy")) return;
  frameBlob = await grab();
  chunks = []; recorder.start(); talk.classList.add("rec"); talk.firstChild.textContent = "Listening…";
});
const stop = () => {
  if (!recorder || recorder.state !== "recording") return;
  recorder.onstop = () => send(frameBlob, new Blob(chunks, { type: recorder.mimeType }), null);
  recorder.stop(); talk.classList.remove("rec");
};
talk.addEventListener("pointerup", stop);
talk.addEventListener("pointerleave", stop);

$("go").onclick = async () => {
  const t = $("text").value.trim();
  if (!t) return;
  if (!pendingFile && !video.videoWidth) { $("file").click(); return; }
  send(await grab(), null, t);
};
$("text").addEventListener("keydown", (e) => e.key === "Enter" && $("go").click());
$("file").onchange = (e) => {
  pendingFile = e.target.files[0];
  if (!pendingFile) return;
  still.src = URL.createObjectURL(pendingFile); viewer.classList.add("frozen");
  $("hint").textContent = "Photo loaded. Type or say what to buy.";
};

async function send(img, audio, text) {
  talk.classList.add("busy"); talk.firstChild.textContent = "Finding it…";
  $("results").innerHTML = ""; $("checkout").classList.add("hidden"); document.querySelector(".err")?.remove();
  const fd = new FormData();
  fd.append("image", img, "frame.jpg");
  if (audio) fd.append("audio", audio, "cmd." + (audio.type.includes("mp4") ? "mp4" : "webm"));
  if (text) fd.append("text", text);
  try {
    const r = await fetch("/api/buy", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    last = j; render(j);
  } catch (e) {
    $("results").insertAdjacentHTML("beforebegin", `<div class="err">${e.message}</div>`);
  } finally {
    talk.classList.remove("busy"); talk.firstChild.textContent = "Hold to talk";
  }
}

function drawBoxes(j) {
  const [fw, fh] = j.frame_size || [1280, 720];
  const rect = overlay.getBoundingClientRect();
  overlay.width = rect.width * devicePixelRatio; overlay.height = rect.height * devicePixelRatio;
  const s = Math.min(overlay.width / fw, overlay.height / fh), ox = (overlay.width - fw * s) / 2, oy = (overlay.height - fh * s) / 2;
  const g = overlay.getContext("2d"); g.clearRect(0, 0, overlay.width, overlay.height);
  g.font = `${12 * devicePixelRatio}px ui-monospace,monospace`;
  for (const c of j.candidates || []) {
    const on = j.box && c.xyxy.join() === j.box.xyxy.join();
    const [x0, y0, x1, y1] = c.xyxy;
    g.strokeStyle = on ? "#ffd34d" : "rgba(255,255,255,.45)"; g.lineWidth = (on ? 3 : 1) * devicePixelRatio;
    g.strokeRect(ox + x0 * s, oy + y0 * s, (x1 - x0) * s, (y1 - y0) * s);
    if (on) { g.fillStyle = "#ffd34d"; g.fillText(`${c.label} ${c.score.toFixed(2)}`, ox + x0 * s + 4, oy + y0 * s - 6); }
  }
}

function render(j) {
  still.src = j.frame; viewer.classList.add("frozen");
  still.onload = () => drawBoxes(j);
  $("trace").classList.remove("hidden");
  $("heard").textContent = `“${j.command.text}”`;
  $("target").textContent = [j.command.target || "(any product)", j.command.max_price && `≤ $${j.command.max_price}`, j.command.size && `size ${j.command.size}`].filter(Boolean).join(" · ");
  $("scr").textContent = j.screen.found ? "found · rectified" : "not found · raw frame";
  $("grd").textContent = j.box ? `${j.box.label} @ ${j.box.score}` : j.reason;
  $("src").textContent = j.source || "-";
  const tot = Object.values(j.timings_ms).reduce((a, b) => a + b, 0) || 1;
  $("timings").innerHTML = Object.entries(j.timings_ms).map(([k, v]) => `<i style="width:${100 * v / tot}%;background:${COLORS[k]}" title="${k} ${v}ms"></i>`).join("");
  $("timings").insertAdjacentHTML("afterend", "");
  document.querySelector(".tlegend")?.remove();
  $("timings").insertAdjacentHTML("afterend", `<div class="tlegend">${Object.entries(j.timings_ms).map(([k, v]) => `${k} ${v}ms`).join(" · ")} · total ${tot}ms</div>`);
  if (!j.ok) return;
  $("results").innerHTML = `<div class="crop"><img src="${j.crop}"><div>query crop → ${j.products.length} matches<br>score = 0.7·visual + 0.3·text (SigLIP)</div></div>` +
    j.products.map((p, i) => `<button class="card" data-i="${i}"><img src="${p.image}" loading="lazy" referrerpolicy="no-referrer">
      <div class="b"><div class="t">${esc(p.title)}</div><div class="p">${p.currency === "USD" ? "$" : p.currency + " "}${p.price.toFixed(2)}</div>
      <div class="s">${esc(p.retailer)} · ${p.score.toFixed(3)}</div></div></button>`).join("");
  document.querySelectorAll(".card").forEach((c) => (c.onclick = () => choose(+c.dataset.i)));
  choose(0);
}

const esc = (s) => String(s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function choose(i) {
  selected = last.products[i];
  document.querySelectorAll(".card").forEach((c) => c.classList.toggle("on", +c.dataset.i === i));
  $("checkout").classList.remove("hidden"); $("receipt").classList.add("hidden");
  $("sel").innerHTML = `<b>${esc(selected.title)}</b> · $${selected.price.toFixed(2)} · ${esc(selected.retailer)}`;
  if (!cfg.paypal_client_id) { $("paypal-buttons").textContent = "PayPal not configured on server."; return; }
  await loadPayPal();
  $("paypal-buttons").innerHTML = "";
  paypal.Buttons({
    style: { layout: "vertical", color: "gold", shape: "rect", label: "buynow" },
    createOrder: async () => {
      const r = await fetch("/api/orders", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product_id: selected.id, frame_id: last.frame_id }) });
      const j = await r.json(); if (!r.ok) throw new Error(j.detail); return j.id;
    },
    onApprove: async (data) => {
      const r = await fetch(`/api/orders/${data.orderID}/capture`, { method: "POST" });
      const j = await r.json();
      $("receipt").classList.remove("hidden");
      $("receipt").innerHTML = r.ok ? `PAID ${j.amount.currency_code} ${j.amount.value}<br>order ${j.id}<br>capture ${j.capture_id}<br>${j.payer || ""}` : `capture failed: ${j.detail}`;
    },
    onError: (e) => { $("receipt").classList.remove("hidden"); $("receipt").textContent = "PayPal error: " + e; },
  }).render("#paypal-buttons");
}

function loadPayPal() {
  if (ppLoaded) return Promise.resolve();
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = `https://www.paypal.com/sdk/js?client-id=${cfg.paypal_client_id}&currency=USD&intent=capture&components=buttons`;
    s.onload = () => { ppLoaded = true; res(); }; s.onerror = rej;
    document.head.appendChild(s);
  });
}

addEventListener("resize", () => last && drawBoxes(last));
init();
