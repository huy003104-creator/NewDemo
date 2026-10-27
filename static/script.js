const $ = (id) => document.getElementById(id);
const feed = $("cameraFeed");
let selectedCamera = null;
function setRunning(running) {
  $("startCamera").disabled = running; $("startVideo").disabled = running; $("stop").disabled = !running;
  $("feedBadge").textContent = running ? "LIVE / PROCESSING" : "OFFLINE";
  $("feedPlaceholder").style.display = running ? "none" : "flex";
}
async function api(url, options={}) {
  const response = await fetch(url, options);
  if (response.status === 401) { location.href="/login"; throw new Error("Please sign in again."); }
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || "Request failed.");
  return data;
}
function escapeHtml(v){return String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function loc(c){return `${c.building} · Floor ${c.floor} · ${c.room}`;}
async function loadCameras(){
 const data=await api("/api/cameras");
 const root=$("cameraRows");
 root.innerHTML=data.cameras.map(c=>`<tr><td><strong>${escapeHtml(c.camera_code)}</strong></td><td>${escapeHtml(c.name)}</td><td>${escapeHtml(loc(c))}</td><td>${escapeHtml(c.details||"—")}</td><td><button class="secondary compact choose-camera" data-code="${escapeHtml(c.camera_code)}">Select</button></td></tr>`).join("")||'<tr><td colspan="5">No cameras registered.</td></tr>';
 root.querySelectorAll(".choose-camera").forEach(b=>b.addEventListener("click",()=>{selectedCamera=data.cameras.find(c=>c.camera_code===b.dataset.code);$("selectedLocation").textContent=`${selectedCamera.camera_code} — ${selectedCamera.name}: ${loc(selectedCamera)}${selectedCamera.details?" · "+selectedCamera.details:""}`;}));
 if(!selectedCamera && data.cameras.length){selectedCamera=data.cameras[0];$("selectedLocation").textContent=`${selectedCamera.camera_code} — ${selectedCamera.name}: ${loc(selectedCamera)}`;}
 $("cameraCount").textContent=data.cameras.length;
}
function renderEvents(events){const root=$("events");if(!events?.length){root.innerHTML='<p class="muted">No events yet.</p>';return;}root.innerHTML=events.slice(0,25).map(e=>`<div class="event ${escapeHtml(e.level||"")}"><time>${escapeHtml(e.time)}</time><strong>${escapeHtml(e.camera_code||"")}</strong> ${escapeHtml(e.location||"")} — ${escapeHtml(e.message)}</div>`).join("");}
function renderIncidents(items){const root=$("incidentRows");const open=items.filter(i=>i.status!=="Resolved").length;$("openIncidents").textContent=open;if(!items.length){root.innerHTML='<tr><td colspan="7" class="muted">No incidents recorded. AI alerts will appear here for staff review.</td></tr>';return;}root.innerHTML=items.map(i=>`<tr><td>${escapeHtml(new Date(i.created_at).toLocaleString())}</td><td>${escapeHtml(i.camera_code)}</td><td>${escapeHtml(i.location)}</td><td>${escapeHtml(i.description)}</td><td><span class="decision ${i.status==="Resolved"?"authorised":""}">${escapeHtml(i.status)}</span></td><td>${escapeHtml(i.resolved_by||i.acknowledged_by||"Unassigned")}</td><td>${i.status!=="Resolved"?`<button class="secondary compact incident-action" data-id="${i.id}" data-status="${i.status==="Open"?"Acknowledged":"Resolved"}">${i.status==="Open"?"Acknowledge":"Resolve"}</button>`:"—"}</td></tr>`).join("");root.querySelectorAll(".incident-action").forEach(b=>b.addEventListener("click",async()=>{try{await api(`/api/incidents/${b.dataset.id}/status`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:b.dataset.status})});await refreshStatus();}catch(e){alert(e.message);}}));}
async function refreshStatus(){try{const d=await api("/api/status");$("topStatus").textContent=d.error?"MODEL / SYSTEM ERROR":d.running?"MONITORING":"SYSTEM READY";$("status").textContent=d.status||"READY";$("status").className=d.status&&d.status.includes("VIOLENCE")?"":"status-ready";$("people").textContent=d.people??"—";$("source").textContent=d.running?`Source: ${d.source}`:(d.error||"No source selected");renderEvents(d.events);renderIncidents(d.incidents||[]);if(d.running){setRunning(true);$("feedPlaceholder").style.display="none";}else setRunning(false);}catch(e){$("topStatus").textContent="BACKEND DISCONNECTED";}}
async function startCamera(){if(!selectedCamera){alert("Select a camera first.");return;}try{await api("/api/start-camera",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({camera_code:selectedCamera.camera_code,source_index:selectedCamera.source_index})});setRunning(true);refreshStatus();}catch(e){alert(e.message);}}
async function startVideo(){const file=$("videoFile").files[0];if(!file){alert("Choose a test video first.");return;}const form=new FormData();form.append("video",file);form.append("camera_code",selectedCamera?.camera_code||"CAM-01");try{await api("/api/start-video",{method:"POST",body:form});setRunning(true);refreshStatus();}catch(e){alert(e.message);}}
async function stopMonitoring(){try{await api("/api/stop",{method:"POST"});setRunning(false);}catch(e){alert(e.message);}}
$("startCamera").addEventListener("click",startCamera);$("startVideo").addEventListener("click",startVideo);$("stop").addEventListener("click",stopMonitoring);
$("cameraForm").addEventListener("submit",async e=>{e.preventDefault();const form=new FormData(e.currentTarget);const body=Object.fromEntries(form.entries());body.source_index=Number(body.source_index||0);try{await api("/api/cameras",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});e.currentTarget.reset();await loadCameras();await refreshStatus();}catch(err){alert(err.message);}});
$("cameraFeed").addEventListener("load",()=>{if($("feedBadge").textContent!=="OFFLINE")$("feedPlaceholder").style.display="none";});
loadCameras().then(refreshStatus).catch(e=>alert(e.message));setInterval(refreshStatus,2500);
