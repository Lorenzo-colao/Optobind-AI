/* ============================================================================
   OptoBind Studio — Frontend Controller & 3D WebGL Molecular Engine
   ============================================================================ */

(() => {
  "use strict";

  // ── Global State ──────────────────────────────────────────────────────────
  const state = {
    activeTab: "builder",
    workspace: null,
    colab: { connected: false, url: "" },

    // Builder state
    builderMode: "metal_site",
    inspectedInfo: null,
    selectedMetalId: null,
    selectedLigandId: null,
    lastBuiltTarget: null,
    builderPdbText: "",
    builderFormat: "pdb",
    builderViewer: null,

    // Pipeline execution state
    execMode: "colab",
    activeJobId: null,
    jobPollTimer: null,
    jobLogOffset: 0,
    completedRunId: null,

    // Results & 3D Inspector state
    runs: [],
    currentRunId: null,
    currentRunData: null,
    resultsFilter: "all",
    resultsSearch: "",
    sortColumn: "composite_score",
    sortDesc: true,
    selectedVariant: null,
    inspViewer: null,
    inspPdbText: "",
    inspFormat: "pdb",
    showSurface: false,
    lastDnaResult: null,
  };

  // ── Utilities ─────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  function showToast(msg, duration = 3400) {
    const el = $("toast");
    if (!el) return;
    el.textContent = msg;
    el.classList.add("visible");
    clearTimeout(el._timer);
    el._timer = setTimeout(() => el.classList.remove("visible"), duration);
  }

  async function apiFetch(url, options = {}) {
    const opts = { ...options };
    if (opts.body && typeof opts.body === "object") {
      opts.headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
      opts.body = JSON.stringify(opts.body);
    }
    const resp = await fetch(url, opts);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  function parseNum(v) {
    if (v === null || v === undefined) return -Infinity;
    const s = String(v).replace("%", "").trim();
    if (!s || s === "N/A" || s === "None") return -Infinity;
    const n = parseFloat(s);
    return Number.isFinite(n) ? n : -Infinity;
  }

  // ── 3Dmol.js Rendering Engine ─────────────────────────────────────────────
  function ensureViewer(containerId, existing) {
    if (existing) {
      existing.resize();
      return existing;
    }
    const el = $(containerId);
    if (!el || typeof $3Dmol === "undefined") return null;
    return $3Dmol.createViewer(el, {
      backgroundColor: "#090a0d",
      antialias: true,
    });
  }

  function drawMetalCoordinationShapes(viewer, model) {
    const atoms = model.selectedAtoms({});
    const metals = new Set(["ZN", "CA", "MG", "FE", "CU", "MN", "NI", "CO", "CD"]);
    const metalAtoms = atoms.filter(
      (a) => metals.has((a.elem || "").toUpperCase()) || metals.has((a.resn || "").toUpperCase())
    );
    const donorNames = new Set(["SG", "ND1", "NE2", "OD1", "OD2", "OE1", "OE2"]);
    const donorAtoms = atoms.filter((a) => donorNames.has((a.atom || "").trim().toUpperCase()));

    metalAtoms.forEach((m) => {
      donorAtoms.forEach((d) => {
        const dx = m.x - d.x;
        const dy = m.y - d.y;
        const dz = m.z - d.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist <= 2.95) {
          viewer.addCylinder({
            start: { x: m.x, y: m.y, z: m.z },
            end: { x: d.x, y: d.y, z: d.z },
            radius: 0.07,
            dashed: true,
            dashLength: 0.18,
            gapLength: 0.14,
            color: "#38bdf8",
            fromCap: 1,
            toCap: 1,
          });
          viewer.addLabel(`${dist.toFixed(2)} Å`, {
            position: { x: (m.x + d.x) / 2, y: (m.y + d.y) / 2, z: (m.z + d.z) / 2 },
            fontSize: 10,
            fontColor: "#e2e8f0",
            backgroundColor: "#111318",
            backgroundOpacity: 0.78,
            borderThickness: 0,
          });
        }
      });
    });
  }

  function renderBuilderStructure(pdbText, format = "pdb", filename = "") {
    if (!pdbText) return;
    state.builderPdbText = pdbText;
    state.builderFormat = format;
    $("builder-mol-empty").style.display = "none";
    if (filename) $("builder-viewer-filename").textContent = filename;

    state.builderViewer = ensureViewer("builder-mol-canvas", state.builderViewer);
    const v = state.builderViewer;
    if (!v) return;

    v.removeAllModels();
    v.removeAllShapes();
    v.removeAllLabels();
    v.removeAllSurfaces();

    const model = v.addModel(pdbText, format);
    const styleMode = $("builder-style-select").value;

    const metals = ["ZN", "CA", "MG", "FE", "CU", "MN", "NI", "CO", "CD"];

    if (styleMode === "sticks") {
      v.setStyle({}, { stick: { radius: 0.16, colorscheme: "Jmol" } });
    } else {
      // Protéine en ruban sobre + chaînes latérales des résidus coordinants/poche
      v.setStyle({ hetflag: false }, { cartoon: { color: "#64748b", thickness: 0.28 } });
      const protAtoms = model.selectedAtoms({ hetflag: false });
      const resCount = new Set(protAtoms.map((a) => `${a.chain}:${a.resi}`)).size;
      if (resCount <= 12) {
        v.setStyle(
          { hetflag: false },
          {
            stick: { radius: 0.18, colorscheme: "orangeCarbon" },
            cartoon: { color: "#64748b", opacity: 0.5 },
          }
        );
        // Étiqueter chaque résidu du motif
        const caAtoms = model.selectedAtoms({ hetflag: false, atom: "CA" });
        caAtoms.forEach((ca) => {
          v.addLabel(`${ca.resn}${ca.resi}`, {
            position: { x: ca.x, y: ca.y, z: ca.z },
            fontSize: 10.5,
            fontColor: "#fcd34d",
            backgroundColor: "#14171f",
            backgroundOpacity: 0.8,
          });
        });
      } else {
        v.addStyle(
          { resn: ["CYS", "HIS", "ASP", "GLU"] },
          { stick: { radius: 0.14, colorscheme: "orangeCarbon" } }
        );
      }
    }

    // Ligands organiques en vert émeraude net (sans invert global qui écraserait la protéine)
    v.setStyle({ hetflag: true }, { stick: { radius: 0.2, colorscheme: "greenCarbon" } });
    v.setStyle({ resn: ["HOH", "WAT"] }, {});

    // Ions métalliques en sphère van der Waals bleue acier
    metals.forEach((m) => {
      v.setStyle({ elem: m }, { sphere: { scale: 0.55, color: "#38bdf8" } });
      v.setStyle({ resn: m }, { sphere: { scale: 0.55, color: "#38bdf8" } });
    });

    if (styleMode === "surface") {
      v.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.55, color: "#1e293b" }, { hetflag: false });
    }

    drawMetalCoordinationShapes(v, model);
    v.zoomTo();
    v.render();
  }

  function renderInspectorStructure(pdbText, format = "pdb") {
    if (!pdbText) return;
    state.inspPdbText = pdbText;
    state.inspFormat = format;
    $("insp-mol-empty").style.display = "none";

    state.inspViewer = ensureViewer("insp-mol-canvas", state.inspViewer);
    const v = state.inspViewer;
    if (!v) return;

    v.removeAllModels();
    v.removeAllShapes();
    v.removeAllLabels();
    v.removeAllSurfaces();

    const model = v.addModel(pdbText, format);
    const colorMode = $("insp-color-mode").value;

    const plddtColor = (atom) => {
      const b = atom.b || 0;
      if (b >= 90) return "#3b82f6";
      if (b >= 70) return "#38bdf8";
      if (b >= 50) return "#f59e0b";
      return "#ef4444";
    };

    if (colorMode === "plddt") {
      v.setStyle({ hetflag: false }, { cartoon: { colorfunc: plddtColor } });
    } else if (colorMode === "spectrum") {
      v.setStyle({ hetflag: false }, { cartoon: { color: "spectrum" } });
    } else {
      v.setStyle({ hetflag: false }, { cartoon: { color: "#94a3b8" } });
    }

    // Mettre en évidence les cystéines et histidines de coordination
    v.addStyle(
      { hetflag: false, resn: ["CYS", "HIS"] },
      { stick: { radius: 0.14, colorscheme: "Jmol" } }
    );

    // Ligands & Métaux (sans invert global qui écraserait le ruban protéique !)
    v.setStyle({ hetflag: true }, { stick: { radius: 0.22, colorscheme: "greenCarbon" } });
    v.setStyle({ resn: ["HOH", "WAT"] }, {});
    ["ZN", "CA", "MG", "FE", "CU", "MN", "NI", "CO", "CD"].forEach((m) => {
      v.setStyle({ elem: m }, { sphere: { scale: 0.58, color: "#38bdf8" } });
      v.setStyle({ resn: m }, { sphere: { scale: 0.58, color: "#38bdf8" } });
    });

    if (state.showSurface) {
      v.addSurface($3Dmol.SurfaceType.MS, { opacity: 0.35, color: "#334155" }, { hetflag: false });
    }

    drawMetalCoordinationShapes(v, model);
    v.zoomTo();
    v.render();
  }

  // ── Navigation Tabs ───────────────────────────────────────────────────────
  function switchTab(tabName) {
    state.activeTab = tabName;
    document.querySelectorAll(".nav-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabName);
    });
    document.querySelectorAll(".view-panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === `view-${tabName}`);
    });
    setTimeout(() => {
      if (tabName === "builder" && state.builderViewer) {
        state.builderViewer.resize();
        state.builderViewer.render();
      }
      if (tabName === "results" && state.inspViewer) {
        state.inspViewer.resize();
        state.inspViewer.render();
      }
    }, 40);
  }

  // ── Workspace Initialization ──────────────────────────────────────────────
  async function loadWorkspace() {
    try {
      const ws = await apiFetch("/api/workspace");
      state.workspace = ws;
      populateInputSelects(ws);
      applyConfigToForm(ws.default_config);
      updateColabBadge(ws.colab);

      if (!ws.colab?.connected && state.execMode === "colab") {
        // Garder Colab sélectionnable mais informer clairement
      }
    } catch (err) {
      showToast(`Erreur initialisation : ${err.message}`);
    }
  }

  function populateInputSelects(ws) {
    const tgtSel = $("cfg-target-select");
    const ligSel = $("cfg-ligand-select");
    const prevTgt = tgtSel.value;
    const prevLig = ligSel.value;

    tgtSel.innerHTML = "";
    (ws.targets || []).forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t.path;
      opt.textContent = `${t.name} (${t.n_residues} AA)`;
      tgtSel.appendChild(opt);
    });
    if (prevTgt && [...tgtSel.options].some((o) => o.value === prevTgt)) {
      tgtSel.value = prevTgt;
    } else if ([...tgtSel.options].some((o) => o.value === "inputs/targets/zn2+_1a1l_auto.pdb")) {
      tgtSel.value = "inputs/targets/zn2+_1a1l_auto.pdb";
    } else if ([...tgtSel.options].some((o) => o.value === "inputs/targets/1a1l_metal_site.pdb")) {
      tgtSel.value = "inputs/targets/1a1l_metal_site.pdb";
    } else if ([...tgtSel.options].some((o) => o.value === "inputs/targets/zn2+_site.pdb")) {
      tgtSel.value = "inputs/targets/zn2+_site.pdb";
    }

    ligSel.innerHTML = "";
    (ws.ligands || []).forEach((l) => {
      const opt = document.createElement("option");
      opt.value = l.path;
      opt.textContent = l.name;
      ligSel.appendChild(opt);
    });
    if (prevLig && [...ligSel.options].some((o) => o.value === prevLig)) {
      ligSel.value = prevLig;
    }
  }

  function applyConfigToForm(cfg) {
    if (!cfg) return;
    const d = cfg.design || {};
    const rf = cfg.rfdiffusion || {};
    const mp = cfg.proteinmpnn || {};
    const pf = cfg.pre_filter || {};
    const af = cfg.alphafold || {};
    const vl = cfg.validation || {};

    if (d.n_variants !== undefined) $("cfg-n-variants").value = d.n_variants;
    if (d.min_length !== undefined) $("cfg-min-len").value = d.min_length;
    if (d.max_length !== undefined) $("cfg-max-len").value = d.max_length;
    if (d.pH !== undefined) $("cfg-ph").value = d.pH;

    if (rf.contigs !== undefined) $("cfg-contigs").value = rf.contigs;
    if (rf.num_steps !== undefined) $("cfg-diff-steps").value = rf.num_steps;
    if (rf.seed !== undefined) $("cfg-seed").value = rf.seed;

    if (mp.model !== undefined) $("cfg-mpnn-model").value = mp.model;
    if (mp.num_seq_per_target !== undefined) $("cfg-seqs-per-scaf").value = mp.num_seq_per_target;
    if (mp.sampling_temp !== undefined) $("cfg-temp").value = mp.sampling_temp;

    if (pf.top_k_per_scaffold !== undefined) $("cfg-topk").value = pf.top_k_per_scaffold;
    if (af.num_trunk_recycles !== undefined) $("cfg-recycles").value = af.num_trunk_recycles;
    if (af.num_diffn_samples !== undefined) $("cfg-samples").value = af.num_diffn_samples;

    if (vl.plddt_threshold !== undefined) $("cfg-plddt-thresh").value = vl.plddt_threshold;
    if (vl.rmsd_threshold !== undefined) $("cfg-rmsd-thresh").value = vl.rmsd_threshold;
    if (vl.motif_rmsd_threshold !== undefined) $("cfg-mrmsd-thresh").value = vl.motif_rmsd_threshold;
  }

  function collectFormConfig() {
    const base = structuredClone(state.workspace?.default_config || {});
    base.inputs = {
      target: $("cfg-target-select").value,
      ligand: $("cfg-ligand-select").value,
    };
    base.design = {
      ...(base.design || {}),
      n_variants: parseInt($("cfg-n-variants").value, 10) || 2,
      min_length: parseInt($("cfg-min-len").value, 10) || 70,
      max_length: parseInt($("cfg-max-len").value, 10) || 100,
      pH: parseFloat($("cfg-ph").value) || 7.4,
    };
    base.rfdiffusion = {
      ...(base.rfdiffusion || {}),
      contigs: $("cfg-contigs").value.trim() || "auto",
      num_steps: parseInt($("cfg-diff-steps").value, 10) || 50,
      seed: parseInt($("cfg-seed").value, 10) || 42,
    };
    base.proteinmpnn = {
      ...(base.proteinmpnn || {}),
      model: $("cfg-mpnn-model").value,
      num_seq_per_target: parseInt($("cfg-seqs-per-scaf").value, 10) || 8,
      sampling_temp: parseFloat($("cfg-temp").value) || 0.1,
      seed: parseInt($("cfg-seed").value, 10) || 42,
    };
    base.pre_filter = {
      ...(base.pre_filter || {}),
      enabled: true,
      top_k_per_scaffold: parseInt($("cfg-topk").value, 10) || 3,
    };
    base.alphafold = {
      ...(base.alphafold || {}),
      backend: "chai-1",
      num_trunk_recycles: parseInt($("cfg-recycles").value, 10) || 3,
      num_diffn_samples: parseInt($("cfg-samples").value, 10) || 3,
    };
    base.validation = {
      ...(base.validation || {}),
      plddt_threshold: parseFloat($("cfg-plddt-thresh").value) || 85.0,
      rmsd_threshold: parseFloat($("cfg-rmsd-thresh").value) || 2.0,
      motif_rmsd_threshold: parseFloat($("cfg-mrmsd-thresh").value) || 1.5,
    };
    return base;
  }

  // ── Google Colab Connection Modal ─────────────────────────────────────────
  function updateColabBadge(colab) {
    state.colab = colab || { connected: false, url: "" };
    const dot = $("colab-dot");
    const lbl = $("colab-label");
    dot.className = "status-dot";

    if (colab && colab.connected) {
      dot.classList.add("online");
      const vramGb = colab.vram_total_mb ? `${(colab.vram_total_mb / 1024).toFixed(1)} GB` : "";
      lbl.textContent = `Colab : ${colab.gpu_name || "GPU"} ${vramGb ? "· " + vramGb : ""} (${colab.latency_ms || 0}ms)`;
      if ($("colab-url-input") && !$("colab-url-input").value) {
        $("colab-url-input").value = colab.url || "";
      }
    } else {
      lbl.textContent = "Google Colab : Déconnecté";
    }
  }

  async function connectColab() {
    const url = $("colab-url-input").value.trim();
    const token = $("colab-token-input").value.trim();
    const telem = $("colab-modal-telemetry");
    const btn = $("btn-test-colab");
    if (!url) {
      showToast("Saisissez l'URL du Worker Google Colab.");
      return;
    }

    btn.disabled = true;
    btn.textContent = "Connexion en cours...";
    telem.style.display = "block";
    telem.textContent = "Interrogation du Worker GPU distant...";

    try {
      const info = await apiFetch("/api/colab/connect", {
        method: "POST",
        body: { url, token },
      });
      updateColabBadge(info);
      const w = info.weights || {};
      telem.innerHTML = [
        `Statut      : CONNECTE (${info.latency_ms} ms)`,
        `GPU         : ${info.gpu_name} (${info.vram_free_mb}/${info.vram_total_mb} MB libres)`,
        `Poids prêts : RFdiffusion=${w.rfdiffusion ? "OUI" : "NON"} | LigandMPNN=${w.ligandmpnn ? "OUI" : "NON"} | ProteinMPNN=${w.proteinmpnn ? "OUI" : "NON"}`,
      ].join("\n");
      showToast(`Connecté à ${info.gpu_name}`);
    } catch (err) {
      updateColabBadge({ connected: false, url });
      telem.textContent = `Échec de connexion : ${err.message}`;
      showToast(`Connexion refusée : ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Connecter & Vérifier le GPU";
    }
  }

  // ── View 1: Automated PDB Target Builder ──────────────────────────────────
  async function inspectPdbSource() {
    const src = $("pdb-source-input").value.trim();
    if (!src) return;
    const btn = $("btn-inspect-pdb");
    btn.disabled = true;
    btn.textContent = "Analyse...";

    try {
      const data = await apiFetch("/api/pdb/inspect", {
        method: "POST",
        body: { source: src },
      });
      state.inspectedInfo = data;
      state.selectedMetalId = data.metals?.[0]?.id || null;
      state.selectedLigandId = data.ligands?.[0]?.id || null;

      $("inspect-summary-badge").textContent = `${data.chains.length} ch · ${data.metals.length} métaux · ${data.ligands.length} lig`;
      renderEntityList();
      renderBuilderStructure(data.pdb_text, "pdb", `${src} (source brute)`);

      const cleanSlug = src.split("/").pop().replace(/\.(pdb|cif|mmcif)$/i, "").toLowerCase();
      $("builder-out-name").value = `${cleanSlug}_${state.builderMode}.pdb`;
    } catch (err) {
      showToast(`Inspection échouée : ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Inspecter";
    }
  }

  function renderEntityList() {
    const listEl = $("builder-entity-list");
    const info = state.inspectedInfo;
    if (!info) return;

    listEl.innerHTML = "";

    if (state.builderMode === "metal_site") {
      if (!info.metals.length) {
        listEl.innerHTML = `<div style="color:var(--crimson);font-size:12px;">Aucun ion métallique détecté dans cette structure. Choisissez « Poche Ligand » ou « Domaine Cible ».</div>`;
        return;
      }
      info.metals.forEach((m) => {
        const div = document.createElement("div");
        div.className = `entity-item ${state.selectedMetalId === m.id ? "selected" : ""}`;
        const coordSummary = m.coordinating_residues.map((c) => `${c.res_name}${c.res_seq}`).join(", ");
        div.innerHTML = `
          <div>
            <div class="mono" style="font-weight:600;color:var(--text-primary);">${m.id} · ${m.geometry}</div>
            <div style="font-size:11px;color:var(--text-muted);">${coordSummary || "Aucun résidu coordinant proche"}</div>
          </div>
          <span class="badge badge-blue">${m.n_coordinating} liaisons</span>
        `;
        div.addEventListener("click", () => {
          state.selectedMetalId = m.id;
          renderEntityList();
        });
        listEl.appendChild(div);
      });
    } else if (state.builderMode === "ligand_pocket") {
      if (!info.ligands.length) {
        listEl.innerHTML = `<div style="color:var(--amber);font-size:12px;">Aucun ligand organique détecté dans ce PDB.</div>`;
        return;
      }
      info.ligands.forEach((l) => {
        const div = document.createElement("div");
        div.className = `entity-item ${state.selectedLigandId === l.id ? "selected" : ""}`;
        div.innerHTML = `
          <div>
            <div class="mono" style="font-weight:600;color:var(--text-primary);">${l.id} (${l.n_heavy_atoms} atomes lourds)</div>
            <div style="font-size:11px;color:var(--text-muted);">Poche : ${l.pocket_residues.slice(0, 5).join(", ")}</div>
          </div>
          <span class="badge badge-neutral">${l.n_pocket_residues} rés.</span>
        `;
        div.addEventListener("click", () => {
          state.selectedLigandId = l.id;
          renderEntityList();
        });
        listEl.appendChild(div);
      });
    } else {
      info.chains.forEach((c) => {
        const div = document.createElement("div");
        const isSel = $("binder-chain-input").value.trim().toUpperCase() === c.chain;
        div.className = `entity-item ${isSel ? "selected" : ""}`;
        div.innerHTML = `
          <div>
            <div class="mono" style="font-weight:600;color:var(--text-primary);">Chaîne ${c.chain} (${c.residue_range})</div>
            <div class="mono" style="font-size:10.5px;color:var(--text-muted);">${c.sequence_preview}</div>
          </div>
          <span class="badge badge-neutral">${c.n_residues} AA</span>
        `;
        div.addEventListener("click", () => {
          $("binder-chain-input").value = c.chain;
          renderEntityList();
        });
        listEl.appendChild(div);
      });
    }
  }

  async function buildTargetPdb() {
    const src = $("pdb-source-input").value.trim();
    if (!src) {
      showToast("Indiquez une source PDB ou UniProt.");
      return;
    }
    const btn = $("btn-build-pdb");
    btn.disabled = true;
    btn.textContent = "Synthèse géométrique 3D en cours...";

    try {
      const payload = {
        source: src,
        mode: state.builderMode,
        output_name: $("builder-out-name").value.trim(),
        metal_id: state.selectedMetalId,
        ligand_id: state.selectedLigandId,
        reporter_ligand: state.builderMode === "metal_site" ? $("reporter-ligand-input").value.trim() : "",
        reporter_distance: parseFloat($("reporter-dist-input").value) || 8.5,
        target_chain: $("binder-chain-input").value.trim(),
        residue_range: $("binder-range-input").value.trim(),
        min_length: parseInt($("builder-min-len").value, 10) || 70,
        max_length: parseInt($("builder-max-len").value, 10) || 100,
      };

      const res = await apiFetch("/api/pdb/build", { method: "POST", body: payload });
      state.lastBuiltTarget = res;

      $("build-result-card").style.display = "block";
      $("built-contigs-output").value = res.recommended_contigs;

      renderBuilderStructure(res.pdb_text, "pdb", res.output_pdb);
      renderGapDiagnostics(res);
      await loadWorkspace();
      showToast(`Cible générée : ${res.output_pdb}`);
    } catch (err) {
      showToast(`Erreur de génération : ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Générer le PDB Cible & Calculer les Contigs";
    }
  }

  function renderGapDiagnostics(res) {
    const geomBadge = $("builder-geom-badge");
    const wrap = $("builder-gap-table-wrap");

    if (res.coordination_info && res.coordination_info.geometry) {
      geomBadge.textContent = `${res.coordination_info.metal} · ${res.coordination_info.geometry} · Angle moyen ${res.coordination_info.mean_angle_deg}°`;
    } else {
      geomBadge.textContent = `Mode : ${res.mode}`;
    }

    const gaps = res.gap_analysis || [];
    if (!gaps.length) {
      wrap.innerHTML = `<div class="mono" style="font-size:11.5px; color:var(--text-secondary);">Contig : ${res.recommended_contigs}</div>`;
      return;
    }

    let html = `<table class="data-table" style="font-size:11.5px;">
      <thead>
        <tr>
          <th>Segment Motif</th>
          <th>Distance Cα–Cα 3D</th>
          <th>Continuité Native</th>
          <th>Boucle RFdiffusion Insérée</th>
        </tr>
      </thead>
      <tbody>`;
    gaps.forEach((g) => {
      html += `<tr>
        <td class="mono" style="color:var(--text-primary);">${g.from} -&gt; ${g.to}</td>
        <td class="mono">${g.ca_distance_angstrom} Å</td>
        <td>${g.contiguous ? '<span class="badge badge-pass">Peptidique</span>' : '<span class="badge badge-neutral">Disjoint 3D</span>'}</td>
        <td class="mono" style="color:#34d399;">${g.inserted_loop}</td>
      </tr>`;
    });
    html += `</tbody></table>`;
    wrap.innerHTML = html;
  }

  // ── View 2: Pipeline Execution & Live Log Stream ──────────────────────────
  function setStepperState(stage, status) {
    const order = ["scaffolding", "sequencing", "prefilter", "validation", "reporting"];
    const normalized = stage === "solubility" ? "reporting" : stage;
    const curIdx = order.indexOf(normalized);

    document.querySelectorAll("#pipeline-stepper .step-node").forEach((node, idx) => {
      node.classList.remove("active", "done");
      if (status === "completed") {
        node.classList.add("done");
      } else if (curIdx >= 0) {
        if (idx < curIdx) node.classList.add("done");
        else if (idx === curIdx && status === "running") node.classList.add("active");
      }
    });
  }

  function appendTerminalLines(lines) {
    if (!lines || !lines.length) return;
    const term = $("job-terminal");
    if (state.jobLogOffset === 0) term.innerHTML = "";

    const frag = document.createDocumentFragment();
    lines.forEach((line) => {
      const div = document.createElement("div");
      if (line.includes("PASS") || line.includes("terminé avec succès")) div.className = "log-line-pass";
      else if (line.includes("ERROR") || line.includes("FAIL ")) div.className = "log-line-err";
      else if (line.includes("WARN")) div.className = "log-line-warn";
      else if (line.includes("MODULE ") || line.includes("====")) div.className = "log-line-head";
      div.textContent = line;
      frag.appendChild(div);
    });
    term.appendChild(frag);
    term.scrollTop = term.scrollHeight;
  }

  async function startPipelineJob() {
    if (state.execMode === "colab" && !state.colab?.connected) {
      $("colab-modal").classList.add("open");
      showToast("Connectez d'abord l'URL de votre Worker Google Colab (ou choisissez Local / Dry-Run).");
      return;
    }

    const cfg = collectFormConfig();
    $("btn-start-job").disabled = true;
    $("btn-cancel-job").style.display = "inline-flex";
    $("btn-open-completed-run").style.display = "none";
    $("job-terminal").textContent = "Initialisation du pipeline...";
    state.jobLogOffset = 0;

    try {
      const res = await apiFetch("/api/jobs/start", {
        method: "POST",
        body: {
          execution_mode: state.execMode,
          config: cfg,
          target: $("cfg-target-select").value,
          ligand: $("cfg-ligand-select").value,
        },
      });
      state.activeJobId = res.run_id;
      $("job-id-label").textContent = `${res.run_id} (${res.mode})`;
      $("job-status-badge").className = "badge badge-blue";
      $("job-status-badge").textContent = "EN COURS";
      setStepperState("scaffolding", "running");
      pollActiveJob();
    } catch (err) {
      $("btn-start-job").disabled = false;
      $("btn-cancel-job").style.display = "none";
      $("job-status-badge").className = "badge badge-fail";
      $("job-status-badge").textContent = "ERREUR";
      $("job-terminal").textContent = `Erreur au démarrage : ${err.message}`;
      showToast(err.message);
    }
  }

  function pollActiveJob() {
    clearTimeout(state.jobPollTimer);
    if (!state.activeJobId) return;

    apiFetch(`/api/jobs/${state.activeJobId}?offset=${state.jobLogOffset}`)
      .then(async (data) => {
        appendTerminalLines(data.logs);
        state.jobLogOffset = data.next_offset;
        $("job-progress-fill").style.width = `${data.progress_pct || 0}%`;
        $("job-elapsed").textContent = `${data.elapsed_sec}s`;
        setStepperState(data.stage, data.status);

        if (data.status === "completed") {
          $("job-status-badge").className = "badge badge-pass";
          $("job-status-badge").textContent = "TERMINE";
          $("btn-start-job").disabled = false;
          $("btn-cancel-job").style.display = "none";
          state.completedRunId = state.activeJobId;
          $("btn-open-completed-run").style.display = "inline-flex";
          await loadRunsList(state.completedRunId);
          showToast(`Run ${state.completedRunId} terminé avec succès.`);
          return;
        }
        if (data.status === "failed" || data.status === "cancelled") {
          $("job-status-badge").className = "badge badge-fail";
          $("job-status-badge").textContent = data.status === "cancelled" ? "INTERROMPU" : "ECHEC";
          $("btn-start-job").disabled = false;
          $("btn-cancel-job").style.display = "none";
          return;
        }
        state.jobPollTimer = setTimeout(pollActiveJob, 900);
      })
      .catch(() => {
        state.jobPollTimer = setTimeout(pollActiveJob, 1500);
      });
  }

  // ── View 3: Results Explorer, 3D Structure & Codon Optimizer ──────────────
  async function loadRunsList(preferredRunId = null) {
    try {
      const data = await apiFetch("/api/runs");
      state.runs = data.runs || [];
      const sel = $("runs-select");
      sel.innerHTML = "";

      if (!state.runs.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "Aucun run disponible";
        sel.appendChild(opt);
        return;
      }

      // Privilégier un vrai run GPU Colab s'il existe au premier chargement
      let targetId = preferredRunId || state.currentRunId;
      if (!targetId) {
        const gpuRun = state.runs.find((r) => r.origin.includes("Colab") && r.n_validated > 0);
        targetId = gpuRun ? gpuRun.run_id : state.runs[0].run_id;
      }

      state.runs.forEach((r) => {
        const opt = document.createElement("option");
        opt.value = r.run_id;
        opt.textContent = `${r.run_id} · ${r.origin} (${r.n_validated} val, max ${r.best_plddt})`;
        sel.appendChild(opt);
      });

      if ([...sel.options].some((o) => o.value === targetId)) {
        sel.value = targetId;
      }
      await loadRunDetail(sel.value);
    } catch (err) {
      showToast(`Erreur lecture runs : ${err.message}`);
    }
  }

  async function loadRunDetail(runId) {
    if (!runId) return;
    state.currentRunId = runId;
    try {
      const detail = await apiFetch(`/api/runs/${runId}`);
      state.currentRunData = detail;
      const s = detail.summary;

      $("kpi-origin").textContent = s.origin;
      $("kpi-total").textContent = String(s.n_total);
      $("kpi-passed").textContent = `${s.n_passed} / ${s.n_validated}`;
      $("kpi-plddt").textContent = s.best_plddt > 0 ? s.best_plddt.toFixed(1) : "—";
      $("kpi-score").textContent = s.best_score > 0 ? `${s.best_score.toFixed(1)}` : "—";

      renderVariantsTable();

      // Sélectionner automatiquement le meilleur candidat validé
      const firstValid = detail.variants.find((v) => v.has_predicted_pdb) || detail.variants[0];
      if (firstValid) {
        selectVariant(firstValid);
      }
    } catch (err) {
      showToast(`Erreur chargement ${runId} : ${err.message}`);
    }
  }

  function renderVariantsTable() {
    const tbody = $("variants-tbody");
    tbody.innerHTML = "";
    if (!state.currentRunData) return;

    let rows = [...state.currentRunData.variants];

    if (state.resultsFilter === "validated") {
      rows = rows.filter((r) => r.plddt_mean && r.plddt_mean !== "N/A");
    } else if (state.resultsFilter === "passed") {
      rows = rows.filter((r) => r.passed === "True");
    }

    if (state.resultsSearch) {
      const q = state.resultsSearch.toLowerCase();
      rows = rows.filter((r) => (r.variant_name || "").toLowerCase().includes(q));
    }

    const col = state.sortColumn;
    rows.sort((a, b) => {
      if (col === "variant_name" || col === "passed") {
        const cmp = String(a[col] || "").localeCompare(String(b[col] || ""));
        return state.sortDesc ? -cmp : cmp;
      }
      const na = parseNum(a[col]);
      const nb = parseNum(b[col]);
      return state.sortDesc ? nb - na : na - nb;
    });

    rows.forEach((r) => {
      const tr = document.createElement("tr");
      if (state.selectedVariant && state.selectedVariant.variant_name === r.variant_name) {
        tr.classList.add("selected");
      }

      const shortName = (r.variant_name || "").replace(`${state.currentRunId}_`, "");
      let statusBadge = `<span class="badge badge-neutral">PRE-FILTRE</span>`;
      if (r.passed === "True") statusBadge = `<span class="badge badge-pass">PASS</span>`;
      else if (r.passed === "False") statusBadge = `<span class="badge badge-fail">FAIL</span>`;

      tr.innerHTML = `
        <td class="mono" style="color:var(--text-primary);font-weight:500;">${shortName}</td>
        <td>${statusBadge}</td>
        <td class="mono" style="font-weight:600;color:var(--text-primary);">${r.composite_score || "—"}</td>
        <td class="mono">${r.plddt_mean || "—"}</td>
        <td class="mono">${r.plddt_pocket || "—"}</td>
        <td class="mono">${r.iptm || "—"}</td>
        <td class="mono">${r.motif_rmsd_angstrom || "—"}</td>
        <td class="mono">${r.rmsd_angstrom || "—"}</td>
        <td class="mono">${r.delta_sasa || "—"}</td>
        <td class="mono">${r.hydrophobic_sasa_pct || "—"}</td>
        <td class="mono">${r.n_interface_hbonds || "—"}</td>
        <td class="mono">${r.ramachandran_outliers_pct || "—"}</td>
        <td class="mono">${r.n_steric_clashes || "—"}</td>
        <td class="mono">${r.solubility_score || "—"}</td>
      `;
      tr.addEventListener("click", () => selectVariant(r));
      tbody.appendChild(tr);
    });
  }

  async function selectVariant(variant) {
    state.selectedVariant = variant;
    state.lastDnaResult = null;
    $("dna-output-meta").style.display = "none";
    $("dna-output-box").style.display = "none";
    $("btn-download-dna").style.display = "none";

    renderVariantsTable();

    const shortTitle = (variant.variant_name || "").replace(`${state.currentRunId}_`, "");
    $("insp-variant-title").textContent = shortTitle;
    $("insp-variant-title").title = variant.variant_name || "";
    $("insp-prot-seq").textContent = variant.sequence || "Séquence non disponible";

    const rejBanner = $("insp-rejection-banner");
    if (String(variant.predicted_pdb || "").startsWith("EXCLUDED:")) {
      rejBanner.style.display = "block";
      rejBanner.textContent = `Écarté par Smart Pre-Filter CPU : ${variant.predicted_pdb.replace("EXCLUDED:", "").trim()}`;
    } else {
      rejBanner.style.display = "none";
    }

    const metrics = [
      ["Score Global", `${variant.composite_score || "0.0"} / 100`],
      ["pLDDT Moyen", variant.plddt_mean || "N/A"],
      ["pLDDT Poche", variant.plddt_pocket || "N/A"],
      ["ipTM / pTM", `${variant.iptm || "—"} / ${variant.ptm || "—"}`],
      ["RMSD Motif", variant.motif_rmsd_angstrom ? `${variant.motif_rmsd_angstrom} Å` : "N/A"],
      ["RMSD Global", variant.rmsd_angstrom ? `${variant.rmsd_angstrom} Å` : "N/A"],
      ["ΔSASA Interface", variant.delta_sasa ? `${variant.delta_sasa} Å²` : "N/A"],
      ["% Surface Hydro", variant.hydrophobic_sasa_pct || "N/A"],
      ["H-Bonds (Dens.)", `${variant.n_interface_hbonds || "0"} (${variant.hbond_density || "0"})`],
      ["SS (H / E / C)", `${variant.helix_pct || "—"} / ${variant.beta_pct || "—"} / ${variant.coil_pct || "—"}`],
      ["pI / Charge 7.4", `${variant.pi || "—"} / ${variant.charge_ph7 || "—"}`],
      ["Protein-Sol", variant.solubility_score !== "N/A" ? `${variant.solubility_score} (${variant.solubility_percent})` : "N/A"],
    ];

    const grid = $("insp-metrics-grid");
    grid.innerHTML = metrics
      .map(
        ([lbl, val]) => `
      <div class="metric-cell">
        <div class="metric-cell-label">${lbl}</div>
        <div class="metric-cell-val">${val}</div>
      </div>`
      )
      .join("");

    await loadVariantStructure3D();
  }

  async function loadVariantStructure3D() {
    if (!state.selectedVariant || !state.currentRunId) return;
    const kind = $("insp-struct-kind").value;
    const coordBadge = $("insp-coord-badge");
    try {
      const sdata = await apiFetch(
        `/api/runs/${state.currentRunId}/structure/${encodeURIComponent(state.selectedVariant.variant_name)}?kind=${kind}`
      );
      if (sdata.metal_coordination && sdata.metal_coordination.metals_count > 0) {
        coordBadge.style.display = "inline-flex";
        coordBadge.textContent = `Métal : ${sdata.metal_coordination.coordination_number} liaisons`;
      } else {
        coordBadge.style.display = "none";
      }
      renderInspectorStructure(sdata.pdb_text, sdata.format);
    } catch {
      $("insp-mol-empty").style.display = "flex";
      coordBadge.style.display = "none";
    }
  }

  // ── DNA Codon Optimization ────────────────────────────────────────────────
  async function optimizeSelectedDna() {
    if (!state.selectedVariant || !state.selectedVariant.sequence) {
      showToast("Sélectionnez d'abord un candidat disposant d'une séquence.");
      return;
    }
    try {
      const res = await apiFetch("/api/dna/optimize", {
        method: "POST",
        body: {
          sequence: state.selectedVariant.sequence,
          variant_name: state.selectedVariant.variant_name,
          host: $("dna-host-select").value,
          add_his_tev: $("dna-his-tag").checked,
        },
      });
      state.lastDnaResult = res;
      const meta = $("dna-output-meta");
      const box = $("dna-output-box");
      meta.style.display = "block";
      meta.textContent = `${res.dna_length_bp} bp · GC ${res.gc_percent}% · Sans sites BsaI/BbsI/EcoRI : ${res.restriction_sites_free ? "OUI" : "NON"}`;
      box.style.display = "block";
      box.textContent = res.dna_sequence;
      $("btn-download-dna").style.display = "inline-flex";
    } catch (err) {
      showToast(`Erreur optimisation ADN : ${err.message}`);
    }
  }

  // ── Event Listeners Binding ───────────────────────────────────────────────
  function bindEvents() {
    // Tabs
    document.querySelectorAll(".nav-tab").forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    // Colab Modal
    $("btn-colab-modal").addEventListener("click", () => $("colab-modal").classList.add("open"));
    $("btn-close-colab-modal").addEventListener("click", () => $("colab-modal").classList.remove("open"));
    $("btn-test-colab").addEventListener("click", connectColab);
    $("btn-disconnect-colab").addEventListener("click", async () => {
      await apiFetch("/api/colab/disconnect", { method: "POST", body: {} });
      updateColabBadge({ connected: false, url: "" });
      $("colab-modal-telemetry").style.display = "none";
      showToast("Déconnecté de Google Colab.");
    });

    // Builder Presets & Inspection
    $("btn-inspect-pdb").addEventListener("click", inspectPdbSource);
    document.querySelectorAll("[data-preset-source]").forEach((chip) => {
      chip.addEventListener("click", () => {
        $("pdb-source-input").value = chip.dataset.presetSource;
        inspectPdbSource();
      });
    });

    $("upload-target-file").addEventListener("change", (ev) => {
      const file = ev.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async () => {
        const b64 = String(reader.result).split(",")[1] || "";
        try {
          const res = await apiFetch("/api/files/upload", {
            method: "POST",
            body: { kind: "target", filename: file.name, content_b64: b64 },
          });
          $("pdb-source-input").value = res.path;
          await loadWorkspace();
          await inspectPdbSource();
        } catch (err) {
          showToast(`Import échoué : ${err.message}`);
        }
      };
      reader.readAsDataURL(file);
    });

    document.querySelectorAll("#builder-mode-cards .mode-card").forEach((card) => {
      card.addEventListener("click", () => {
        document.querySelectorAll("#builder-mode-cards .mode-card").forEach((c) => c.classList.remove("active"));
        card.classList.add("active");
        state.builderMode = card.dataset.mode;
        $("protein-binder-fields").style.display = state.builderMode === "protein_binder" ? "block" : "none";
        $("reporter-fields").style.display = state.builderMode === "metal_site" ? "block" : "none";
        renderEntityList();
      });
    });

    $("btn-fetch-pubchem").addEventListener("click", async () => {
      const q = $("reporter-ligand-input").value.trim();
      if (!q) return;
      try {
        const res = await apiFetch("/api/pdb/pubchem", { method: "POST", body: { query: q } });
        $("reporter-ligand-input").value = res.path;
        await loadWorkspace();
        showToast(`Ligand 3D téléchargé : ${res.name} (${res.n_heavy_atoms} atomes lourds)`);
      } catch (err) {
        showToast(err.message);
      }
    });

    $("btn-build-pdb").addEventListener("click", buildTargetPdb);
    $("builder-style-select").addEventListener("change", () => {
      if (state.builderPdbText) renderBuilderStructure(state.builderPdbText, state.builderFormat);
    });
    $("btn-builder-reset-cam").addEventListener("click", () => {
      if (state.builderViewer) {
        state.builderViewer.zoomTo();
        state.builderViewer.render();
      }
    });

    $("btn-apply-to-pipeline").addEventListener("click", async () => {
      if (!state.lastBuiltTarget) return;
      await loadWorkspace();
      $("cfg-target-select").value = state.lastBuiltTarget.output_pdb;
      if (state.lastBuiltTarget.ligand_sdf) {
        $("cfg-ligand-select").value = state.lastBuiltTarget.ligand_sdf;
      }
      $("cfg-contigs").value = state.lastBuiltTarget.recommended_contigs || "auto";
      switchTab("pipeline");
      showToast("Cible PDB et contigs transférés vers le Pipeline.");
    });

    // Pipeline Execution Mode & Presets
    document.querySelectorAll("#exec-mode-cards .mode-card").forEach((card) => {
      card.addEventListener("click", () => {
        document.querySelectorAll("#exec-mode-cards .mode-card").forEach((c) => c.classList.remove("active"));
        card.classList.add("active");
        state.execMode = card.dataset.exec;
      });
    });

    document.querySelectorAll("[data-cfg-preset]").forEach((chip) => {
      chip.addEventListener("click", () => {
        document.querySelectorAll("[data-cfg-preset]").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        const p = chip.dataset.cfgPreset;
        if (p === "fast") {
          $("cfg-n-variants").value = 1;
          $("cfg-diff-steps").value = 25;
          $("cfg-seqs-per-scaf").value = 4;
          $("cfg-topk").value = 2;
          $("cfg-recycles").value = 1;
          $("cfg-samples").value = 1;
        } else if (p === "balanced") {
          $("cfg-n-variants").value = 2;
          $("cfg-diff-steps").value = 50;
          $("cfg-seqs-per-scaf").value = 8;
          $("cfg-topk").value = 3;
          $("cfg-recycles").value = 2;
          $("cfg-samples").value = 2;
        } else {
          $("cfg-n-variants").value = 4;
          $("cfg-diff-steps").value = 100;
          $("cfg-seqs-per-scaf").value = 12;
          $("cfg-topk").value = 3;
          $("cfg-recycles").value = 3;
          $("cfg-samples").value = 3;
        }
      });
    });

    $("btn-start-job").addEventListener("click", startPipelineJob);
    $("btn-cancel-job").addEventListener("click", async () => {
      if (!state.activeJobId) return;
      await apiFetch(`/api/jobs/${state.activeJobId}/cancel`, { method: "POST", body: {} });
    });
    $("btn-clear-logs").addEventListener("click", () => {
      $("job-terminal").innerHTML = "";
    });
    $("btn-open-completed-run").addEventListener("click", async () => {
      if (state.completedRunId) {
        await loadRunsList(state.completedRunId);
        switchTab("results");
      }
    });

    // Results & Inspector
    $("runs-select").addEventListener("change", (ev) => loadRunDetail(ev.target.value));
    $("btn-refresh-runs").addEventListener("click", () => loadRunsList(state.currentRunId));
    $("results-search").addEventListener("input", (ev) => {
      state.resultsSearch = ev.target.value.trim();
      renderVariantsTable();
    });

    document.querySelectorAll("#results-filter-chips .chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        document.querySelectorAll("#results-filter-chips .chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        state.resultsFilter = chip.dataset.filter;
        renderVariantsTable();
      });
    });

    document.querySelectorAll("#variants-table th[data-sort]").forEach((th) => {
      th.addEventListener("click", () => {
        const col = th.dataset.sort;
        if (state.sortColumn === col) state.sortDesc = !state.sortDesc;
        else {
          state.sortColumn = col;
          state.sortDesc = true;
        }
        renderVariantsTable();
      });
    });

    $("insp-struct-kind").addEventListener("change", loadVariantStructure3D);
    $("insp-color-mode").addEventListener("change", () => {
      if (state.inspPdbText) renderInspectorStructure(state.inspPdbText, state.inspFormat);
    });
    $("btn-insp-surface").addEventListener("click", () => {
      state.showSurface = !state.showSurface;
      $("btn-insp-surface").classList.toggle("active", state.showSurface);
      if (state.inspPdbText) renderInspectorStructure(state.inspPdbText, state.inspFormat);
    });

    $("btn-copy-prot-seq").addEventListener("click", () => {
      if (state.selectedVariant?.sequence) {
        navigator.clipboard.writeText(state.selectedVariant.sequence);
        showToast("Séquence protéique copiée.");
      }
    });

    $("btn-optimize-dna").addEventListener("click", optimizeSelectedDna);
    $("btn-download-dna").addEventListener("click", () => {
      if (!state.lastDnaResult) return;
      const blob = new Blob([state.lastDnaResult.fasta], { type: "text/plain;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${state.selectedVariant.variant_name}_DNA.fasta`;
      a.click();
    });

    ["xlsx", "csv", "zip"].forEach((kind) => {
      $(`btn-dl-${kind}`).addEventListener("click", () => {
        if (!state.currentRunId) return;
        window.open(`/api/runs/${state.currentRunId}/download/${kind}`, "_blank");
      });
    });
  }

  // ── Boot Sequence ─────────────────────────────────────────────────────────
  window.addEventListener("DOMContentLoaded", async () => {
    bindEvents();
    await loadWorkspace();
    await loadRunsList();

    // Charger immédiatement la cible locale de référence en 3D au démarrage
    try {
      const previewPath = state.workspace?.targets?.some((t) => t.name === "zn2+_1a1l_auto.pdb")
        ? "inputs/targets/zn2+_1a1l_auto.pdb"
        : "inputs/targets/zn2+_site.pdb";
      const prev = await apiFetch(`/api/pdb/preview?path=${encodeURIComponent(previewPath)}`);
      renderBuilderStructure(prev.pdb_text, prev.format, prev.filename);
    } catch {
      // Ignore si absent
    }
  });
})();
