const hostInput = document.querySelector("#host");
const modelsInput = document.querySelector("#models");
const suiteInput = document.querySelector("#suite");
const examples = document.querySelector("#examples");
const planView = document.querySelector("#plan-view");
const resultsCard = document.querySelector("#results-card");
const results = document.querySelector("#results");
const cancelView = document.querySelector("#cancel-view");
const saveState = document.querySelector("#save-state");
const confirmBox = document.querySelector("#confirm");
const runButton = document.querySelector("#run");

confirmBox.addEventListener("change", () => {
  runButton.disabled = !confirmBox.checked;
});

async function api(path, body) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderPlan(plan) {
  const rows = (plan.jobs || [])
    .map((job) => {
      const loras = (job.loras || []).map((item) => `${item.name} @ ${item.strength_model}`).join(", ") || "none";
      return `<tr>
        <td>${escapeHtml(job.case_id)}</td>
        <td>${escapeHtml(job.action)}</td>
        <td>${escapeHtml(job.seed)}</td>
        <td>${escapeHtml(job.steps)}</td>
        <td>${escapeHtml(job.sampler)}</td>
        <td>${escapeHtml(job.width)}×${escapeHtml(job.height)}</td>
        <td>${escapeHtml(loras)}</td>
        <td>${escapeHtml(job.estimated_label)}</td>
      </tr>`;
    })
    .join("");
  const warnings = (plan.warnings || []).map((item) => `<p class="callout bad">${escapeHtml(item)}</p>`).join("");
  planView.innerHTML = `
    ${warnings}
    <p class="callout">Queue check: ${escapeHtml(plan.queue_check)}</p>
    <p><strong>${escapeHtml(plan.estimated_label)}</strong> for ${plan.jobs.length} case(s). ${escapeHtml(plan.estimate_note || "")}</p>
    <div class="table-scroll"><table>
      <thead><tr><th>Case</th><th>Action</th><th>Seed</th><th>Steps</th><th>Sampler</th><th>Size</th><th>LoRAs</th><th>Estimate</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

function shot(path, label) {
  if (!path) {
    return `<figure><figcaption>${label}</figcaption><div class="meta">No image</div></figure>`;
  }
  const src = `/media/${path.split("/").map(encodeURIComponent).join("/")}`;
  return `<figure><figcaption>${label}</figcaption><img class="shot" alt="${label}" src="${src}"></figure>`;
}

function renderResults(payload) {
  resultsCard.hidden = false;
  const tone = payload.mode === "completed" ? "good" : "bad";
  const cases = (payload.cases || [])
    .map((item) => {
      const metrics = [
        item.phash == null ? null : `pHash ${item.phash}`,
        item.ssim == null ? null : `SSIM ${Number(item.ssim).toFixed(4)}`,
        item.lpips == null ? null : `LPIPS ${Number(item.lpips).toFixed(4)}`,
      ]
        .filter(Boolean)
        .join(" · ");
      const reasons = []
        .concat(item.reasons || [], item.changes || [])
        .map((reason) => `<li>${escapeHtml(reason)}</li>`)
        .join("");
      return `<article>
        <div class="actions"><h3>${escapeHtml(item.case_id)}</h3><span class="pill ${escapeHtml(item.verdict)}">${escapeHtml(item.verdict)}</span></div>
        <p class="meta">${escapeHtml(metrics || item.error || "")}</p>
        <div class="frames">${shot(item.before_path, "Before")}${shot(item.after_path, "After")}${shot(item.diff_path, "Diff")}</div>
        <ul>${reasons}</ul>
      </article>`;
    })
    .join("");
  const report = payload.report_path
    ? `<p><a href="/media/${payload.report_path.split("/").map(encodeURIComponent).join("/")}">Open HTML report</a></p>`
    : "";
  results.innerHTML = `<p class="callout ${tone}">${escapeHtml(payload.message || payload.mode)}</p>${report}${cases}`;
}

async function saveConfig() {
  const saved = await api("/api/config", {
    comfy_host: hostInput.value.trim(),
    models_root: modelsInput.value.trim(),
  });
  saveState.textContent = saved.comfy_host
    ? `Saved host ${saved.comfy_host} on this computer.`
    : "Saved. No ComfyUI host is set yet.";
}

async function dryRun() {
  planView.innerHTML = `<p class="meta">Planning locally. ComfyUI is not contacted.</p>`;
  const plan = await api("/api/plan", {
    suite_path: suiteInput.value.trim(),
    host: hostInput.value.trim(),
  });
  renderPlan(plan);
}

hostInput.addEventListener("change", () => saveConfig().catch((error) => {
  saveState.textContent = error.message;
}));
modelsInput.addEventListener("change", () => saveConfig().catch((error) => {
  saveState.textContent = error.message;
}));

examples.addEventListener("change", () => {
  suiteInput.value = examples.value;
  dryRun().catch((error) => {
    planView.innerHTML = `<p class="callout bad">${escapeHtml(error.message)}</p>`;
  });
});

document.querySelector("#plan").addEventListener("click", () => {
  dryRun().catch((error) => {
    planView.innerHTML = `<p class="callout bad">${escapeHtml(error.message)}</p>`;
  });
});

runButton.addEventListener("click", async () => {
  if (!confirmBox.checked) {
    return;
  }
  runButton.disabled = true;
  resultsCard.hidden = false;
  results.innerHTML = `<p class="meta">Checking the queue, then rendering. This page waits until ComfyUI finishes. There is no background schedule.</p>`;
  try {
    await saveConfig();
    const payload = await api("/api/run", {
      suite_path: suiteInput.value.trim(),
      host: hostInput.value.trim(),
      models_root: modelsInput.value.trim(),
      confirm_gpu: true,
      update_baseline: document.querySelector("#update").checked,
    });
    if (payload.mode === "dry_run") {
      renderPlan(payload);
    }
    renderResults(payload);
    await dryRun();
  } catch (error) {
    results.innerHTML = `<p class="callout bad">${escapeHtml(error.message)}</p>`;
  } finally {
    runButton.disabled = !confirmBox.checked;
  }
});

document.querySelector("#cancel").addEventListener("click", async () => {
  const agreed = window.confirm(
    "Cancel SeedRegress jobs on this ComfyUI host? Jobs from other programs stay queued."
  );
  if (!agreed) {
    return;
  }
  cancelView.innerHTML = `<p class="meta">Contacting ComfyUI…</p>`;
  try {
    const payload = await api("/api/cancel", {
      confirm_cancel: true,
      host: hostInput.value.trim(),
    });
    cancelView.innerHTML = `<p class="callout">${escapeHtml(payload.message)}</p>`;
  } catch (error) {
    cancelView.innerHTML = `<p class="callout bad">${escapeHtml(error.message)}</p>`;
  }
});

async function boot() {
  const config = await api("/api/config");
  hostInput.value = config.comfy_host || "";
  modelsInput.value = config.models_root || "";
  const listed = await api("/api/examples");
  examples.innerHTML = (listed.examples || [])
    .map((item) => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.name)}</option>`)
    .join("");
  if (listed.examples && listed.examples.length) {
    suiteInput.value = listed.examples[0].path;
    await dryRun();
  }
}

boot().catch((error) => {
  planView.innerHTML = `<p class="callout bad">${escapeHtml(error.message)}</p>`;
});
