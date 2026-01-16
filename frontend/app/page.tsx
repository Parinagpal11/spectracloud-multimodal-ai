"use client";

import { useMemo, useState } from "react";

type AnalyzeResponse = {
  status?: string;
  prediction?: string;
  confidence?: number;
  spectrum_plot_png_base64?: string;

  feature_dim?: number;
  label?: string | null;
  label_display?: string | null;
  prototype_score?: number | null;
  all_scores?: Record<string, number> | null;

  received?: {
    spectrum_filename?: string;
    image_filename?: string;
  };

  detail?: any;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

export default function Page() {
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [imgFile, setImgFile] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState<AnalyzeResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const plotSrc = useMemo(() => {
    if (!res?.spectrum_plot_png_base64) return null;
    return `data:image/png;base64,${res.spectrum_plot_png_base64}`;
  }, [res]);

  const topMatches = useMemo(() => {
    const scores = res?.all_scores;
    if (!scores) return [];
    return Object.entries(scores)
      .filter(([, v]) => typeof v === "number" && Number.isFinite(v))
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
  }, [res]);

  const confidence01 = useMemo(() => {
    const c = res?.confidence;
    if (c == null || !Number.isFinite(c)) return null;
    return clamp01(c);
  }, [res]);

  async function onAnalyze() {
    setErr(null);
    setRes(null);

    if (!csvFile) return setErr("Upload a spectrum CSV.");
    if (!imgFile) return setErr("Upload an image (png/jpg).");

    const fd = new FormData();
    fd.append("spectrum", csvFile);
    fd.append("image", imgFile);

    setLoading(true);
    try {
      const r = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: fd,
      });

      const data = (await r.json()) as AnalyzeResponse;

      if (!r.ok) {
        const d: any = (data as any)?.detail;
        if (Array.isArray(d)) setErr(d.map((e) => e.msg).join(", "));
        else if (typeof d === "string") setErr(d);
        else setErr("Backend error");
        return;
      }

      setRes(data);
    } catch (e: any) {
      setErr(e?.message ?? "Network error");
    } finally {
      setLoading(false);
    }
  }

  const labelText = res?.label_display ?? res?.label ?? "—";
  const protoText =
    res?.prototype_score == null ? "—" : res.prototype_score.toFixed(3);
  const confText = res?.confidence == null ? "—" : res.confidence.toFixed(3);

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Top bar */}
      <div className="border-b border-zinc-800/60 bg-zinc-950/70 backdrop-blur">
        <div className="mx-auto max-w-6xl px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-2xl bg-gradient-to-br from-indigo-500/80 to-fuchsia-500/70 shadow-lg shadow-indigo-500/10" />
            <div>
              <div className="text-lg font-semibold leading-tight">SpectraCloud</div>
              <div className="text-sm text-zinc-400">
                Multimodal demo: Raman spectrum + microscopy image
              </div>
            </div>
          </div>

          <div className="hidden md:flex items-center gap-2 text-xs text-zinc-400">
            <span className="rounded-full border border-zinc-800 px-2 py-1">
              API: {API_BASE}
            </span>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="mx-auto max-w-6xl px-6 py-8">
        <div className="grid gap-6 lg:grid-cols-5">
          {/* Left: Upload card */}
          <section className="lg:col-span-2 rounded-2xl border border-zinc-800/70 bg-zinc-900/30 p-5 shadow-xl shadow-black/20">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-base font-semibold">Run analysis</h2>
                <p className="mt-1 text-sm text-zinc-400">
                  Upload a Raman CSV and a microscopy image. We’ll return label,
                  confidence, top prototype matches, and a plot.
                </p>
              </div>
              <div className="rounded-xl border border-zinc-800 bg-zinc-950/40 px-3 py-1.5 text-xs text-zinc-300">
                Demo-ready
              </div>
            </div>

            <div className="mt-5 grid gap-4">
              {/* Spectrum CSV */}
              <div className="rounded-2xl border border-zinc-800/70 bg-zinc-950/30 p-4">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">Spectrum CSV</div>
                  <div className="text-xs text-zinc-500">Required</div>
                </div>
                <p className="mt-1 text-xs text-zinc-500">
                  Expected headers like wavelength/intensity (or compatible).
                </p>

                <label className="mt-3 block">
                  <input
                    type="file"
                    accept=".csv"
                    className="block w-full text-sm file:mr-4 file:rounded-xl file:border-0 file:bg-zinc-800 file:px-4 file:py-2 file:text-zinc-100 hover:file:bg-zinc-700"
                    onChange={(e) => setCsvFile(e.target.files?.[0] ?? null)}
                  />
                </label>

                {csvFile && (
                  <div className="mt-3 flex items-center justify-between gap-3 rounded-xl border border-zinc-800 bg-zinc-950/40 px-3 py-2">
                    <div className="truncate text-sm text-zinc-200">{csvFile.name}</div>
                    <button
                      onClick={() => setCsvFile(null)}
                      className="text-xs text-zinc-400 hover:text-zinc-200"
                    >
                      Remove
                    </button>
                  </div>
                )}
              </div>

              {/* Image */}
              <div className="rounded-2xl border border-zinc-800/70 bg-zinc-950/30 p-4">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">Microscopy image</div>
                  <div className="text-xs text-zinc-500">Required</div>
                </div>
                <p className="mt-1 text-xs text-zinc-500">
                  PNG/JPG (BloodMNIST works great for demo).
                </p>

                <label className="mt-3 block">
                  <input
                    type="file"
                    accept="image/png,image/jpeg"
                    className="block w-full text-sm file:mr-4 file:rounded-xl file:border-0 file:bg-zinc-800 file:px-4 file:py-2 file:text-zinc-100 hover:file:bg-zinc-700"
                    onChange={(e) => setImgFile(e.target.files?.[0] ?? null)}
                  />
                </label>

                {imgFile && (
                  <div className="mt-3 flex items-center justify-between gap-3 rounded-xl border border-zinc-800 bg-zinc-950/40 px-3 py-2">
                    <div className="truncate text-sm text-zinc-200">{imgFile.name}</div>
                    <button
                      onClick={() => setImgFile(null)}
                      className="text-xs text-zinc-400 hover:text-zinc-200"
                    >
                      Remove
                    </button>
                  </div>
                )}
              </div>

              {/* CTA */}
              <button
                onClick={onAnalyze}
                disabled={loading}
                className="group inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-4 py-3 font-semibold text-white shadow-lg shadow-fuchsia-500/10 hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                    Analyzing…
                  </>
                ) : (
                  <>
                    Analyze
                    <span className="opacity-80 group-hover:translate-x-0.5 transition-transform">
                      →
                    </span>
                  </>
                )}
              </button>

              {/* Error */}
              {err && (
                <div className="rounded-2xl border border-red-900/40 bg-red-950/20 p-4 text-sm text-red-200">
                  <div className="font-semibold">Error</div>
                  <div className="mt-1 opacity-90">{String(err)}</div>
                </div>
              )}
            </div>
          </section>

          {/* Right: Results */}
          <section className="lg:col-span-3 grid gap-6">
            {/* Result summary */}
            <div className="rounded-2xl border border-zinc-800/70 bg-zinc-900/30 p-5 shadow-xl shadow-black/20">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-base font-semibold">Result</h2>
                  <p className="mt-1 text-sm text-zinc-400">
                    Prototype similarity + spectrum preprocessing preview.
                  </p>
                </div>
                {res?.status === "success" ? (
                  <span className="rounded-full border border-emerald-900/40 bg-emerald-950/20 px-3 py-1 text-xs text-emerald-200">
                    Success
                  </span>
                ) : (
                  <span className="rounded-full border border-zinc-800 bg-zinc-950/40 px-3 py-1 text-xs text-zinc-300">
                    Waiting
                  </span>
                )}
              </div>

              <div className="mt-5 grid gap-4 md:grid-cols-3">
                <div className="rounded-2xl border border-zinc-800/70 bg-zinc-950/30 p-4">
                  <div className="text-xs text-zinc-500">Predicted label</div>
                  <div className="mt-1 text-lg font-semibold truncate">{labelText}</div>
                  <div className="mt-2 text-xs text-zinc-500">
                    Feature dim: {res?.feature_dim ?? "—"}
                  </div>
                </div>

                <div className="rounded-2xl border border-zinc-800/70 bg-zinc-950/30 p-4">
                  <div className="text-xs text-zinc-500">Prototype score</div>
                  <div className="mt-1 text-lg font-semibold">{protoText}</div>
                  <div className="mt-2 text-xs text-zinc-500">
                    Higher is better (cosine similarity)
                  </div>
                </div>

                <div className="rounded-2xl border border-zinc-800/70 bg-zinc-950/30 p-4">
                  <div className="flex items-center justify-between">
                    <div className="text-xs text-zinc-500">Confidence</div>
                    <div className="text-xs text-zinc-400">{confText}</div>
                  </div>

                  <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-indigo-400 to-fuchsia-400 transition-all"
                      style={{
                        width:
                          confidence01 == null
                            ? "0%"
                            : `${Math.round(confidence01 * 100)}%`,
                      }}
                    />
                  </div>

                  {confidence01 != null && confidence01 < 0.65 ? (
                    <div className="mt-3 text-xs text-amber-200/90">
                      Low confidence — try a known-good demo spectrum.
                    </div>
                  ) : (
                    <div className="mt-3 text-xs text-zinc-500">
                      Confidence is derived from signal quality + scoring.
                    </div>
                  )}
                </div>
              </div>

              {/* Top matches */}
              <div className="mt-5">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-semibold">Top prototype matches</div>
                  <div className="text-xs text-zinc-500">
                    (Cosine similarity)
                  </div>
                </div>

                {topMatches.length === 0 ? (
                  <div className="mt-3 rounded-2xl border border-zinc-800/70 bg-zinc-950/20 p-4 text-sm text-zinc-400">
                    Run an analysis to see ranked matches.
                  </div>
                ) : (
                  <div className="mt-3 overflow-hidden rounded-2xl border border-zinc-800/70">
                    <div className="divide-y divide-zinc-800/70">
                      {topMatches.map(([k, v], i) => (
                        <div
                          key={k}
                          className="flex items-center justify-between bg-zinc-950/30 px-4 py-3"
                        >
                          <div className="flex items-center gap-3">
                            <div className="flex h-7 w-7 items-center justify-center rounded-xl border border-zinc-800 bg-zinc-950 text-xs text-zinc-300">
                              {i + 1}
                            </div>
                            <div className="text-sm font-medium text-zinc-200">
                              {k}
                            </div>
                          </div>
                          <div className="text-sm text-zinc-200 tabular-nums">
                            {v.toFixed(3)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Received */}
              {res?.received && (
                <div className="mt-5 text-xs text-zinc-500">
                  Received:{" "}
                  <span className="text-zinc-300">
                    {res.received.spectrum_filename ?? "—"}
                  </span>{" "}
                  &{" "}
                  <span className="text-zinc-300">
                    {res.received.image_filename ?? "—"}
                  </span>
                </div>
              )}
            </div>

            {/* Plot */}
            <div className="rounded-2xl border border-zinc-800/70 bg-zinc-900/30 p-5 shadow-xl shadow-black/20">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-base font-semibold">Spectrum plot</div>
                  <div className="text-sm text-zinc-400">
                    Raw vs processed preview
                  </div>
                </div>
              </div>

              {!plotSrc ? (
                <div className="mt-4 rounded-2xl border border-zinc-800/70 bg-zinc-950/20 p-6 text-sm text-zinc-400">
                  No plot yet. Upload files and run analysis.
                </div>
              ) : (
                <div className="mt-4 overflow-hidden rounded-2xl border border-zinc-800/70 bg-zinc-950/20">
                  <img
                    src={plotSrc}
                    alt="Spectrum plot"
                    className="w-full"
                  />
                </div>
              )}
            </div>
          </section>
        </div>
        {/* Results Summary Card */}
<div className="rounded-2xl border border-zinc-800/70 bg-zinc-900/40 p-5 shadow-lg">
  <h3 className="text-base font-semibold">Results Summary</h3>

  {/* Verdict */}
  <p className="mt-3 text-sm text-zinc-200">
    {confidence01 != null && confidence01 >= 0.75
      ? "This sample closely matches a known reference pattern."
      : confidence01 != null && confidence01 >= 0.6
      ? "This sample shows partial similarity to known reference patterns."
      : "This sample does not strongly match known reference patterns."}
  </p>

  {/* Confidence */}
  <div className="mt-4">
    <div className="flex justify-between text-sm">
      <span className="font-medium">Confidence</span>
      <span className="text-zinc-300">
        {res?.confidence != null
          ? `${res.confidence.toFixed(2)}`
          : "—"}
      </span>
    </div>

    <div className="mt-2 h-2 w-full rounded-full bg-zinc-800">
      <div
        className="h-full rounded-full bg-gradient-to-r from-indigo-400 to-fuchsia-400"
        style={{
          width:
            confidence01 == null
              ? "0%"
              : `${Math.round(confidence01 * 100)}%`,
        }}
      />
    </div>

    <div className="mt-2 text-xs text-zinc-400">
      {confidence01 != null && confidence01 >= 0.75
        ? "High confidence result"
        : confidence01 != null && confidence01 >= 0.6
        ? "Moderate confidence result"
        : "Low confidence — review recommended"}
    </div>
  </div>

  {/* Closest Pattern */}
  <div className="mt-4">
    <div className="text-sm font-medium">
      Closest known chemical pattern
    </div>
    <div className="mt-1 text-sm text-zinc-300">
      {res?.label_display ?? res?.label ?? "—"}
    </div>
    <div className="mt-1 text-xs text-zinc-500">
      Based on similarity to reference Raman spectra
    </div>
  </div>

  {/* Explanation */}
  <div className="mt-4 text-sm text-zinc-400">
    The system cleaned the chemical signal, compared its shape and intensity
    against known reference patterns, and estimated reliability based on signal
    quality.
  </div>

  {/* Next Steps */}
  <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-950/40 p-3 text-xs text-zinc-300">
    {confidence01 != null && confidence01 < 0.6
      ? "Recommended action: Re-measure the sample or review manually."
      : "Recommended action: No immediate follow-up required."}
  </div>
</div>


        {/* Footer */}
        <div className="mt-10 text-center text-xs text-zinc-500">
           Spectra + image ingestion • Prototype-based explainability
        </div>
      </div>
    </main>
  );
}
