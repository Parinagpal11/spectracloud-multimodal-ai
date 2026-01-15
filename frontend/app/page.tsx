"use client";

import { useMemo, useState } from "react";

type AnalyzeResponse = {
  status?: string;
  prediction?: string;
  confidence?: number;
  spectrum_plot_png_base64?: string;

  feature_dim?: number;
  label?: string | null;
  prototype_score?: number | null;
  all_scores?: Record<string, number> | null;

  received?: {
    spectrum_filename?: string;
    image_filename?: string;
  };

  detail?: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

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
      .slice(0, 3);
  }, [res]);

  const confidence01 = useMemo(() => {
    const c = res?.confidence;
    if (c == null || !Number.isFinite(c)) return null;
    return Math.max(0, Math.min(1, c));
  }, [res]);

  async function onAnalyze() {
    setErr(null);
    setRes(null);

    if (!csvFile) return setErr("Upload a spectrum CSV.");

    const fd = new FormData();
    fd.append("spectrum", csvFile);

    // ✅ image is optional now
    if (imgFile) fd.append("image", imgFile);

    setLoading(true);
    try {
      const r = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: fd,
      });

      const data = (await r.json()) as AnalyzeResponse;

      if (!r.ok) {
  const d: any = data?.detail;

  if (Array.isArray(d)) {
    // FastAPI validation errors
    setErr(d.map((e) => e.msg).join(", "));
  } else if (typeof d === "string") {
    setErr(d);
  } else {
    setErr("Backend error");
  }
  return;
}


      setRes(data);
    } catch (e: any) {
      setErr(e?.message ?? "Network error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 960, margin: "0 auto", padding: 24 }}>
      <h1 style={{ fontSize: 28, fontWeight: 700 }}>SpectraCloud</h1>
      <p style={{ opacity: 0.8 }}>
        Upload Raman spectrum (image optional) → get label + confidence + plot.
      </p>

      <div style={{ display: "grid", gap: 14, marginTop: 18 }}>
        <div>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Spectrum CSV</div>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setCsvFile(e.target.files?.[0] ?? null)}
          />
          {csvFile && (
            <div style={{ opacity: 0.8, marginTop: 6 }}>{csvFile.name}</div>
          )}
        </div>

        <div>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>
            Image (png/jpg) <span style={{ opacity: 0.7 }}>(optional)</span>
          </div>
          <input
            type="file"
            accept="image/png,image/jpeg"
            onChange={(e) => setImgFile(e.target.files?.[0] ?? null)}
          />
          {imgFile && (
            <div style={{ opacity: 0.8, marginTop: 6 }}>{imgFile.name}</div>
          )}

          {imgFile && (
            <button
              onClick={() => setImgFile(null)}
              style={{
                marginTop: 8,
                padding: "6px 10px",
                borderRadius: 10,
                border: "1px solid rgba(255,255,255,0.2)",
                width: "fit-content",
                cursor: "pointer",
                opacity: 0.9,
              }}
            >
              Remove image
            </button>
          )}
        </div>

        <button
          onClick={onAnalyze}
          disabled={loading}
          style={{
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid rgba(255,255,255,0.2)",
            cursor: loading ? "not-allowed" : "pointer",
            width: "fit-content",
          }}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>

        {err && <div style={{ color: "#ff6b6b", fontWeight: 600 }}>{err}</div>}

        {res && (
          <section style={{ marginTop: 18 }}>
            <h2 style={{ fontSize: 20, fontWeight: 700 }}>Result</h2>

            {/* ✅ confidence bar + warning */}
            <div style={{ marginTop: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <div style={{ fontWeight: 600 }}>Confidence</div>
                <div style={{ opacity: 0.85 }}>
                  {res.confidence == null ? "None" : res.confidence.toFixed(3)}
                </div>
              </div>

              <div
                style={{
                  height: 10,
                  borderRadius: 999,
                  border: "1px solid rgba(255,255,255,0.2)",
                  marginTop: 8,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width:
                      confidence01 == null ? "0%" : `${Math.round(confidence01 * 100)}%`,
                    background: "rgba(255,255,255,0.65)",
                  }}
                />
              </div>

              {confidence01 != null && confidence01 < 0.65 && (
                <div style={{ marginTop: 10, color: "#ffd166", fontWeight: 600 }}>
                  Low confidence — likely noisy / mismatched spectrum. Try a known-good sample.
                </div>
              )}
            </div>

            <div style={{ display: "grid", gap: 8, marginTop: 14 }}>
              <div>
                <b>Label:</b> {res.label ?? "None"}
              </div>
              <div>
                <b>Prototype score:</b>{" "}
                {res.prototype_score == null ? "None" : res.prototype_score.toFixed(3)}
              </div>
              <div>
                <b>Feature dim:</b> {res.feature_dim ?? "?"}
              </div>
            </div>

            {/* ✅ top-3 explainability */}
            {topMatches.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontWeight: 700, marginBottom: 8 }}>Top matches</div>
                <div style={{ display: "grid", gap: 8 }}>
                  {topMatches.map(([k, v]) => (
                    <div
                      key={k}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        padding: "10px 12px",
                        borderRadius: 12,
                        border: "1px solid rgba(255,255,255,0.15)",
                      }}
                    >
                      <div style={{ fontWeight: 600 }}>{k}</div>
                      <div style={{ opacity: 0.9 }}>{v.toFixed(3)}</div>
                    </div>
                  ))}
                </div>
                <div style={{ opacity: 0.7, marginTop: 8, fontSize: 13 }}>
                  (This is cosine similarity vs each class prototype.)
                </div>
              </div>
            )}

            {plotSrc && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontWeight: 600, marginBottom: 8 }}>Spectrum Plot</div>
                <img
                  src={plotSrc}
                  alt="Spectrum plot"
                  style={{
                    width: "100%",
                    borderRadius: 12,
                    border: "1px solid rgba(255,255,255,0.15)",
                  }}
                />
              </div>
            )}
          </section>
        )}
      </div>
    </main>
  );
}
