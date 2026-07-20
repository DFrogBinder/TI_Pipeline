"use strict";

const state = {
  queue: [],
  stats: null,
  busy: false,
  zoom: 100,
  subjects: [],
};

const elements = {};

document.addEventListener("DOMContentLoaded", async () => {
  for (const id of [
    "saveState", "reviewTab", "subjectsTab", "undoButton", "exportButton", "rescanButton",
    "acceptedSubjects", "targetSubjects", "targetStatus", "targetProgressBar",
    "declinedSubjects", "maybeSubjects", "inReviewSubjects", "unreviewedSubjects",
    "remainingImages", "reviewView", "subjectsView", "queueMode", "queueOrder", "firstTissue",
    "reloadQueueButton", "zoomOut", "zoomIn", "zoomReset", "zoomRange", "zoomValue",
    "fullScreenButton", "imageStage", "imageLoading", "reviewImage", "emptyQueue",
    "emptyTitle", "emptyMessage", "queueRemaining", "currentSubject", "currentTissue",
    "currentView", "currentFile", "decisionNote", "acceptButton", "maybeButton",
    "declineButton", "subjectSearch", "subjectTableBody", "toast", "targetDialog",
    "dialogAcceptedCount", "dialogExportButton", "dialogContinueButton",
  ]) {
    elements[id] = document.getElementById(id);
  }

  bindEvents();
  try {
    const bootstrap = await api("/api/bootstrap");
    updateStats(bootstrap.stats);
    populateTissueOptions(bootstrap.tissues || []);
    elements.saveState.textContent = "Saved to SQLite";
    await loadQueue();
  } catch (error) {
    showError(error);
  }
});

function bindEvents() {
  elements.reviewTab.addEventListener("click", () => showView("review"));
  elements.subjectsTab.addEventListener("click", () => showView("subjects"));
  elements.queueMode.addEventListener("change", loadQueue);
  elements.queueOrder.addEventListener("change", loadQueue);
  elements.firstTissue.addEventListener("change", loadQueue);
  elements.reloadQueueButton.addEventListener("click", loadQueue);
  elements.acceptButton.addEventListener("click", () => decide("accept"));
  elements.maybeButton.addEventListener("click", () => decide("maybe"));
  elements.declineButton.addEventListener("click", () => decide("decline"));
  elements.undoButton.addEventListener("click", undoLast);
  elements.exportButton.addEventListener("click", exportResults);
  elements.rescanButton.addEventListener("click", rescanFolder);
  elements.dialogExportButton.addEventListener("click", exportResults);
  elements.dialogContinueButton.addEventListener("click", () => elements.targetDialog.close());
  elements.subjectSearch.addEventListener("input", renderSubjectTable);
  elements.zoomRange.addEventListener("input", () => setZoom(Number(elements.zoomRange.value)));
  elements.zoomOut.addEventListener("click", () => setZoom(state.zoom - 10));
  elements.zoomIn.addEventListener("click", () => setZoom(state.zoom + 10));
  elements.zoomReset.addEventListener("click", () => setZoom(100));
  elements.fullScreenButton.addEventListener("click", async () => {
    try {
      await elements.imageStage.requestFullscreen();
    } catch (error) {
      showError(error);
    }
  });
  elements.reviewImage.addEventListener("load", () => elements.imageLoading.classList.add("hidden"));
  elements.reviewImage.addEventListener("error", () => {
    elements.imageLoading.textContent = "The image could not be loaded.";
    elements.imageLoading.classList.remove("hidden");
  });
  document.addEventListener("keydown", (event) => {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || state.busy || elements.targetDialog.open) return;
    if (event.key.toLowerCase() === "a") decide("accept");
    if (event.key.toLowerCase() === "m") decide("maybe");
    if (event.key.toLowerCase() === "d") decide("decline");
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

async function loadQueue() {
  setBusy(true);
  elements.imageLoading.textContent = "Loading review queue…";
  elements.imageLoading.classList.remove("hidden");
  elements.reviewImage.classList.add("hidden");
  elements.emptyQueue.classList.add("hidden");
  try {
    const mode = elements.queueMode.value;
    const order = elements.queueOrder.value;
    const parameters = new URLSearchParams({ mode, order });
    if (elements.firstTissue.value) {
      parameters.set("first_tissue", elements.firstTissue.value);
    }
    const payload = await api(`/api/queue?${parameters.toString()}`);
    state.queue = payload.items;
    renderCurrent();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

function currentItem() { return state.queue.length ? state.queue[0] : null; }

function renderCurrent() {
  const item = currentItem();
  elements.queueRemaining.textContent = String(state.queue.length);
  elements.decisionNote.value = item?.note || "";
  if (!item) {
    elements.reviewImage.removeAttribute("src");
    elements.reviewImage.classList.add("hidden");
    elements.imageLoading.classList.add("hidden");
    elements.emptyQueue.classList.remove("hidden");
    elements.emptyTitle.textContent = elements.queueMode.value === "maybe" ? "Maybe pass complete" : "New-image queue complete";
    elements.emptyMessage.textContent = elements.queueMode.value === "maybe"
      ? "No more maybe images remain in this pass. Reload the queue to revisit retained maybes."
      : "All currently eligible images have a decision.";
    elements.currentSubject.textContent = "No image selected";
    elements.currentTissue.textContent = "—";
    elements.currentView.textContent = "—";
    elements.currentFile.textContent = "—";
    setDecisionButtons(false);
    return;
  }
  elements.emptyQueue.classList.add("hidden");
  elements.imageLoading.textContent = "Loading image…";
  elements.imageLoading.classList.remove("hidden");
  elements.reviewImage.classList.remove("hidden");
  elements.reviewImage.src = item.image_url;
  elements.reviewImage.alt = `${item.subject_id}, ${item.tissue_display}, ${item.view} view`;
  elements.currentSubject.textContent = item.subject_id;
  elements.currentTissue.textContent = item.tissue_display;
  elements.currentView.textContent = titleCase(item.view);
  elements.currentFile.textContent = item.relative_path;
  setDecisionButtons(true);
}

async function decide(decision) {
  const item = currentItem();
  if (!item || state.busy) return;
  setBusy(true);
  elements.saveState.textContent = "Saving…";
  try {
    const result = await api("/api/decision", {
      method: "POST",
      body: JSON.stringify({
        image_id: item.id,
        decision,
        note: elements.decisionNote.value,
      }),
    });
    if (decision === "decline") {
      state.queue = state.queue.filter((queued) => queued.subject_id !== item.subject_id);
      showToast(`${item.subject_id} declined and removed from the remaining queue.`);
    } else {
      state.queue.shift();
    }
    updateStats(result.stats);
    elements.saveState.textContent = "Saved to SQLite";
    renderCurrent();
    if (result.target_just_reached) {
      elements.dialogAcceptedCount.textContent = String(result.stats.accepted_subjects);
      elements.targetDialog.showModal();
    }
  } catch (error) {
    elements.saveState.textContent = "Save failed";
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function undoLast() {
  if (state.busy) return;
  setBusy(true);
  try {
    const result = await api("/api/undo", { method: "POST", body: "{}" });
    updateStats(result.stats);
    showToast(`Restored the previous decision for ${result.subject_id || "the last image"}.`);
    await loadQueue();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function rescanFolder() {
  if (state.busy) return;
  setBusy(true);
  elements.saveState.textContent = "Scanning…";
  try {
    const result = await api("/api/rescan", { method: "POST", body: "{}" });
    updateStats(result.stats);
    populateTissueOptions(result.tissues || []);
    elements.saveState.textContent = "Saved to SQLite";
    showToast(`Found ${result.scan.images} images across ${result.scan.subjects} subjects; skipped ${result.scan.skipped}.`);
    await loadQueue();
  } catch (error) {
    elements.saveState.textContent = "Scan failed";
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function exportResults() {
  if (state.busy) return;
  setBusy(true);
  try {
    const result = await api("/api/export", { method: "POST", body: "{}" });
    updateStats(result.stats);
    showToast(`Exports updated in ${result.files.directory}`);
    const link = document.createElement("a");
    link.href = "/api/export/accepted_subjects.txt";
    link.download = "accepted_subjects.txt";
    link.click();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

function updateStats(stats) {
  state.stats = stats;
  elements.acceptedSubjects.textContent = stats.accepted_subjects;
  elements.targetSubjects.textContent = stats.target;
  elements.declinedSubjects.textContent = stats.declined_subjects;
  elements.maybeSubjects.textContent = stats.maybe_subjects;
  elements.inReviewSubjects.textContent = stats.in_review_subjects;
  elements.unreviewedSubjects.textContent = stats.unreviewed_subjects;
  elements.remainingImages.textContent = stats.remaining_new_images;
  elements.targetStatus.textContent = stats.target_reached ? "Target reached" : "Target in progress";
  const progress = Math.min(100, (stats.accepted_subjects / stats.target) * 100);
  elements.targetProgressBar.style.width = `${progress}%`;
}

function populateTissueOptions(tissues) {
  const selected = elements.firstTissue.value;
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "Default tag order";
  const tissueOptions = tissues.map((tissue) => {
    const option = document.createElement("option");
    option.value = tissue.slug;
    option.textContent = tissue.display;
    return option;
  });
  elements.firstTissue.replaceChildren(defaultOption, ...tissueOptions);
  if (tissues.some((tissue) => tissue.slug === selected)) {
    elements.firstTissue.value = selected;
  }
}

function setBusy(busy) {
  state.busy = busy;
  for (const button of [
    elements.acceptButton, elements.maybeButton, elements.declineButton,
    elements.undoButton, elements.exportButton, elements.rescanButton,
    elements.reloadQueueButton,
  ]) {
    button.disabled = busy || ([elements.acceptButton, elements.maybeButton, elements.declineButton].includes(button) && !currentItem());
  }
}

function setDecisionButtons(enabled) {
  for (const button of [elements.acceptButton, elements.maybeButton, elements.declineButton]) {
    button.disabled = !enabled || state.busy;
  }
}

function setZoom(value) {
  state.zoom = Math.max(50, Math.min(250, value));
  elements.zoomRange.value = state.zoom;
  elements.zoomValue.textContent = `${state.zoom}%`;
  elements.reviewImage.style.maxWidth = `${state.zoom}%`;
  elements.reviewImage.style.maxHeight = `${state.zoom}%`;
}

async function showView(view) {
  const subjects = view === "subjects";
  elements.reviewView.classList.toggle("hidden", subjects);
  elements.subjectsView.classList.toggle("hidden", !subjects);
  elements.reviewTab.classList.toggle("active", !subjects);
  elements.subjectsTab.classList.toggle("active", subjects);
  if (subjects) {
    try {
      const payload = await api("/api/subjects");
      state.subjects = payload.subjects;
      renderSubjectTable();
    } catch (error) {
      showError(error);
    }
  }
}

function renderSubjectTable() {
  const query = elements.subjectSearch.value.trim().toLowerCase();
  const rows = state.subjects.filter((row) => row.subject_id.toLowerCase().includes(query));
  elements.subjectTableBody.replaceChildren(...rows.map((row) => {
    const tr = document.createElement("tr");
    const values = [
      row.subject_id,
      row.status,
      row.accepted_images,
      row.maybe_images,
      row.declined_images,
      row.unreviewed_images,
      row.total_images,
    ];
    for (const [index, value] of values.entries()) {
      const td = document.createElement("td");
      if (index === 1) {
        const pill = document.createElement("span");
        pill.className = `status-pill ${row.status}`;
        pill.textContent = titleCase(row.status.replace("_", " "));
        td.appendChild(pill);
      } else {
        td.textContent = String(value);
      }
      tr.appendChild(td);
    }
    return tr;
  }));
}

let toastTimer = null;
function showToast(message, error = false) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", error);
  elements.toast.classList.remove("hidden");
  toastTimer = setTimeout(() => elements.toast.classList.add("hidden"), 4500);
}

function showError(error) { showToast(error.message || String(error), true); }
function titleCase(value) { return value.replace(/\b\w/g, (char) => char.toUpperCase()); }
