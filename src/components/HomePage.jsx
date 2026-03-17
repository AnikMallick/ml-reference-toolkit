// Home.jsx  —  ML Reference Library Homepage
// Drop into src/components/Home.jsx and wire via App.jsx routes shown at bottom of file.

import { useState } from "react";
import { Link } from "react-router-dom";
import Header from "./Header";
// ─────────────────────────────────────────────────────────────────────────────
//  MUTED DARK DESIGN TOKENS
//  Copy the CSS block below into your global stylesheet (src/index.css) so
//  every page in the app inherits the same palette.
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
//  TOOL REGISTRY  (7 reference tools + the playbook)
// ─────────────────────────────────────────────────────────────────────────────
const TOOLS = [
  {
    index: "00",
    route: "/playbook",
    title: "Playbook",
    subtitle: "End-to-end workflow guide",
    accent: "var(--accent-amber)",
    fill:   "var(--fill-amber)",
    badge:  "Start here",
    badgeStyle: { bg:"rgba(122,104,68,0.18)", color:"#b09060", border:"rgba(122,104,68,0.35)" },
    count:  "16 phases",
    icon:   "🧭",
    desc:   "Step-by-step operational guide from raw data to a deployed model. Each phase has code snippets, pitfall warnings, and cross-references to every other table.",
    topics: ["Problem framing & metric choice", "CV strategy by data structure", "Feature engineering sequence", "Baseline → iteration → ensembling", "Adversarial validation", "Competition submission strategy"],
  },
  {
    index: "01",
    route: "/eda",
    title: "EDA & Visualisation",
    subtitle: "What it detects · what to do · who benefits",
    accent: "var(--accent-teal)",
    fill:   "var(--fill-teal)",
    badge:  "Step 1",
    badgeStyle: { bg:"rgba(46,112,112,0.18)", color:"#3ea0a0", border:"rgba(46,112,112,0.35)" },
    count:  "60 techniques",
    icon:   "🔭",
    desc:   "Every EDA technique mapped to what it detects, the action to take when you find it, which models benefit, and direct cross-references into the FE and Problems tables.",
    topics: ["Automated EDA (ydata-profiling, SweetViz)", "Univariate & bivariate analysis", "Missing data patterns (missingno)", "Adversarial validation", "SHAP summary & dependence plots", "Time series, NLP, image EDA"],
  },
  {
    index: "02",
    route: "/features",
    title: "Feature Engineering",
    subtitle: "Encoding · transforms · selection · tests",
    accent: "var(--accent-sage)",
    fill:   "var(--fill-sage)",
    badge:  "Step 2",
    badgeStyle: { bg:"rgba(58,110,88,0.18)", color:"#50a078", border:"rgba(58,110,88,0.35)" },
    count:  "81 techniques",
    icon:   "⚗️",
    desc:   "Comprehensive coverage of categorical encoding, numeric transforms, tabular feature creation, NLP, image, and time series feature engineering, statistical tests, and feature selection.",
    topics: ["8 categorical encoding strategies", "Numeric transforms & scaling", "Interaction, ratio, datetime features", "Text & image feature extraction", "Time series (tsfresh, Catch22)", "Statistical tests & feature selection"],
  },
  {
    index: "03",
    route: "/sklearn",
    title: "sklearn Model Reference",
    subtitle: "When to use · when not to · real-world usage",
    accent: "var(--accent-steel)",
    fill:   "var(--fill-steel)",
    badge:  "Step 3",
    badgeStyle: { bg:"rgba(56,88,120,0.18)", color:"#5090c0", border:"rgba(56,88,120,0.35)" },
    count:  "55+ models",
    icon:   "📐",
    desc:   "Every sklearn estimator with description, use-when / don't-use guidance, and real-world industry applications — organised by model family with filterable categories.",
    topics: ["Linear models & regularisation", "SVMs · Naive Bayes · Neighbors", "Ensemble methods (RF, GBM, HGBT)", "Clustering & dimensionality reduction", "Neural nets · semi-supervised", "Outlier detection & calibration"],
  },
  {
    index: "04",
    route: "/problems",
    title: "ML Problems Reference",
    subtitle: "Detect · root cause · fix",
    accent: "var(--accent-rose)",
    fill:   "var(--fill-rose)",
    badge:  "Debug",
    badgeStyle: { bg:"rgba(110,56,80,0.18)", color:"#b06080", border:"rgba(110,56,80,0.35)" },
    count:  "41 problems",
    icon:   "🩺",
    desc:   "When something goes wrong — target leakage, overfitting, class imbalance, distribution shift, training instability — find the root cause, diagnostic steps, and concrete fix.",
    topics: ["All 3 forms of data leakage", "Overfitting · underfitting", "Class imbalance & threshold tuning", "CV strategy failures", "Distribution & concept drift", "Competition-specific pitfalls"],
  },
  {
    index: "05",
    route: "/torch",
    title: "PyTorch & Skorch",
    subtitle: "Deep learning · sklearn integration",
    accent: "var(--accent-violet)",
    fill:   "var(--fill-violet)",
    badge:  "Deep Learning",
    badgeStyle: { bg:"rgba(90,72,136,0.18)", color:"#9878d8", border:"rgba(90,72,136,0.35)" },
    count:  "84 entries",
    icon:   "🔥",
    desc:   "Complete PyTorch reference covering tensors, layers, activations, loss functions, optimizers, schedulers, regularisation, and data pipelines — plus Skorch for sklearn-compatible DL training.",
    topics: ["PyTorch core (tensors, autograd, AMP)", "Layers, activations, loss functions", "Optimizers & LR schedulers", "Custom Dataset & DataLoader", "Skorch NeuralNet + sklearn Pipeline", "Architecture patterns & transfer learning"],
  },
  {
    index: "06",
    route: "/deploy",
    title: "MLOps & Deployment",
    subtitle: "Serve · monitor · govern · automate",
    accent: "var(--accent-seafoam)",
    fill:   "var(--fill-seafoam)",
    badge:  "Production",
    badgeStyle: { bg:"rgba(46,104,104,0.18)", color:"#40a0a0", border:"rgba(46,104,104,0.35)" },
    count:  "37 entries",
    icon:   "🚀",
    desc:   "The full post-training lifecycle: serialization, experiment tracking, model registry, REST APIs, containerization, CI/CD, drift detection, deployment strategies, and retraining governance.",
    topics: ["joblib · ONNX · TorchScript · MLflow Registry", "FastAPI · BentoML · Triton serving", "Docker · Kubernetes · CI/CD with GitHub Actions", "Evidently · NannyML · Prometheus monitoring", "Blue-green · Canary · Shadow deployment", "Retraining triggers · Model cards · Anti-patterns"],
  },
];

// ─────────────────────────────────────────────────────────────────────────────
//  WORKFLOW SPINE  (how the tables connect)
// ─────────────────────────────────────────────────────────────────────────────
const FLOW = [
  { label: "Understand",  sub: "Playbook 00–01",  accent: "var(--accent-amber)"  },
  { label: "EDA",         sub: "EDA table",        accent: "var(--accent-teal)"   },
  { label: "Engineer",    sub: "FE table",          accent: "var(--accent-sage)"   },
  { label: "Model",       sub: "sklearn / PyTorch", accent: "var(--accent-steel)"  },
  { label: "Debug",       sub: "Problems table",    accent: "var(--accent-rose)"   },
  { label: "Deploy",      sub: "MLOps table",       accent: "var(--accent-seafoam)"},
  { label: "Monitor",     sub: "Playbook 14–15",    accent: "var(--accent-violet)" },
];

// ─────────────────────────────────────────────────────────────────────────────
//  DECISION GUIDE
// ─────────────────────────────────────────────────────────────────────────────
const DECISIONS = [
  { q: "Starting a new project or competition?",         a: "→ Playbook", r: "/playbook", accent: "var(--accent-amber)"  },
  { q: "Want to understand your dataset?",               a: "→ EDA table", r: "/eda",     accent: "var(--accent-teal)"   },
  { q: "Need to encode a categorical column?",           a: "→ FE table", r: "/features", accent: "var(--accent-sage)"   },
  { q: "Choosing a model for the task?",                 a: "→ sklearn table", r: "/sklearn", accent: "var(--accent-steel)" },
  { q: "CV score suspiciously high?",                    a: "→ Problems → 'leakage'", r: "/problems", accent: "var(--accent-rose)" },
  { q: "Building a neural network?",                     a: "→ PyTorch table", r: "/torch", accent: "var(--accent-violet)" },
  { q: "Ready to put a model in production?",            a: "→ MLOps table", r: "/deploy", accent: "var(--accent-seafoam)" },
  { q: "Model degrading in production?",                 a: "→ MLOps → Drift Detection", r: "/deploy", accent: "var(--accent-seafoam)" },
  { q: "CV doesn't match test performance?",             a: "→ Problems → 'CV strategy'", r: "/problems", accent: "var(--accent-rose)" },
  { q: "Want sklearn API for a PyTorch model?",          a: "→ PyTorch → Skorch", r: "/torch", accent: "var(--accent-violet)" },
  { q: "Need SHAP feature importance?",                  a: "→ EDA → SHAP plots", r: "/eda", accent: "var(--accent-teal)" },
  { q: "Competition final submission?",                  a: "→ Playbook Phase 15", r: "/playbook", accent: "var(--accent-amber)" },
];

// ─────────────────────────────────────────────────────────────────────────────
//  STATS
// ─────────────────────────────────────────────────────────────────────────────
const STATS = [
  { n: "384+",  label: "techniques & entries" },
  { n: "7",     label: "reference tables"      },
  { n: "16",    label: "workflow phases"        },
  { n: "15",    label: "MLOps categories"       },
];

// ─────────────────────────────────────────────────────────────────────────────
//  TOOL CARD COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
function ToolCard({ tool, animDelay }) {
  const [hover, setHover] = useState(false);
  const [open,  setOpen]  = useState(false);

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        background:    hover ? "var(--bg-elevated)" : "var(--bg-surface)",
        border:        `1px solid ${hover ? "var(--border-strong)" : "var(--border-subtle)"}`,
        borderLeft:    `3px solid ${tool.accent}`,
        borderRadius:  "var(--r-lg)",
        padding:       "22px 22px 18px",
        display:       "flex",
        flexDirection: "column",
        gap:           "12px",
        transition:    "background var(--t-base), border-color var(--t-base), transform var(--t-base)",
        transform:     hover ? "translateY(-1px)" : "translateY(0)",
        animation:     `fadeUp 0.45s ease ${animDelay}s both`,
      }}
    >
      {/* Header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "10px" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "10px", minWidth: 0 }}>
          <span style={{
            fontFamily: "var(--font-mono)", fontSize: "0.65rem",
            color: "var(--text-dim)", letterSpacing: "0.06em", flexShrink: 0,
          }}>{tool.index}</span>
          <span style={{
            fontFamily: "var(--font-display)", fontSize: "1.22rem",
            fontWeight: 500, color: "var(--text-primary)",
            letterSpacing: "-0.01em", lineHeight: 1.2,
          }}>
            {tool.icon} {tool.title}
          </span>
        </div>
        <span style={{
          padding: "2px 9px", borderRadius: "var(--r-sm)",
          fontSize: "0.62rem", fontWeight: 700, letterSpacing: "0.05em",
          background: tool.badgeStyle.bg, color: tool.badgeStyle.color,
          border: `1px solid ${tool.badgeStyle.border}`,
          whiteSpace: "nowrap", flexShrink: 0, marginTop: "2px",
        }}>{tool.badge}</span>
      </div>

      {/* Subtitle + count */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "0.76rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
          {tool.subtitle}
        </span>
        <span style={{
          marginLeft: "auto", fontFamily: "var(--font-mono)", fontSize: "0.64rem",
          color: tool.accent, background: tool.fill,
          padding: "2px 8px", borderRadius: "var(--r-sm)", whiteSpace: "nowrap",
          flexShrink: 0,
        }}>{tool.count}</span>
      </div>

      {/* Description */}
      <p style={{ margin: 0, fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>
        {tool.desc}
      </p>

      {/* Expandable topics */}
      <div>
        <button
          onClick={() => setOpen(o => !o)}
          style={{
            background: "none", border: "none", padding: 0, cursor: "pointer",
            fontSize: "0.7rem", color: "var(--text-tertiary)",
            display: "flex", alignItems: "center", gap: "5px",
          }}
        >
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem" }}>
            {open ? "▲" : "▶"}
          </span>
          {open ? "hide contents" : "what's inside"}
        </button>
        {open && (
          <ul style={{ margin: "10px 0 0", padding: 0, listStyle: "none" }}>
            {tool.topics.map(t => (
              <li key={t} style={{
                fontSize: "0.76rem", color: "var(--text-secondary)",
                lineHeight: 1.9, display: "flex", alignItems: "center", gap: "8px",
              }}>
                <span style={{
                  width: "4px", height: "4px", borderRadius: "50%",
                  background: tool.accent, flexShrink: 0, display: "inline-block",
                }} />
                {t}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Navigation button */}
      <Link
        to={tool.route}
        style={{
          marginTop: "auto",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "9px 16px", borderRadius: "var(--r-md)",
          background: hover ? tool.fill : "transparent",
          border: `1px solid ${hover ? tool.accent : "var(--border-default)"}`,
          color: hover ? tool.accent : "var(--text-tertiary)",
          fontSize: "0.76rem", transition: "all var(--t-fast)",
        }}
      >
        <span>Open {tool.title}</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem" }}>↗</span>
      </Link>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  WORKFLOW NODE
// ─────────────────────────────────────────────────────────────────────────────
function FlowNode({ step, index, isLast }) {
  return (
    <div style={{ display: "flex", alignItems: "flex-start", flex: isLast ? "0 0 auto" : "1 1 0", minWidth: 0 }}>
      <div style={{ textAlign: "center", minWidth: "80px" }}>
        {/* Circle */}
        <div style={{
          width: "34px", height: "34px", borderRadius: "50%",
          background: `color-mix(in srgb, ${step.accent} 14%, var(--bg-surface))`,
          border: `1px solid color-mix(in srgb, ${step.accent} 38%, transparent)`,
          display: "flex", alignItems: "center", justifyContent: "center",
          margin: "0 auto 8px",
          fontFamily: "var(--font-mono)", fontSize: "0.65rem",
          color: step.accent, fontWeight: 600,
        }}>
          {String(index + 1).padStart(2, "0")}
        </div>
        <div style={{
          fontSize: "0.7rem", fontWeight: 600,
          color: "var(--text-primary)", marginBottom: "3px",
        }}>{step.label}</div>
        <div style={{ fontSize: "0.6rem", color: "var(--text-tertiary)", lineHeight: 1.4 }}>
          {step.sub}
        </div>
      </div>
      {!isLast && (
        <div style={{
          flex: 1, height: "1px", background: "var(--border-subtle)",
          margin: "0 4px", position: "relative", top: "-9px",
          minWidth: "8px",
        }} />
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  HOME PAGE
// ─────────────────────────────────────────────────────────────────────────────
export default function Home() {
  return (
    <div style={{ fontFamily: "var(--font-body)", background: "var(--bg-base)", minHeight: "100vh", color: "var(--text-primary)" }}>

      {/* ── Inject CSS variables & keyframes ── */}
      <style>{`
@keyframes fadeUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to   { opacity: 1; }
        }

        .home-grid {
          display: grid;
          gap: 12px;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        }

        /* 3 columns on wide screens, last row balanced */
        @media (min-width: 960px) {
          .home-grid {
            grid-template-columns: repeat(3, 1fr);
          }
        }

        .flow-row {
          display: flex;
          align-items: flex-start;
          overflow-x: auto;
          padding-bottom: 4px;
          gap: 0;
        }

        .decision-grid {
          display: grid;
          gap: 1px;
          background: var(--border-subtle);
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          border-radius: var(--r-lg);
          overflow: hidden;
          border: 1px solid var(--border-subtle);
        }

        /* Nav hover */
        .top-nav a {
          font-size: 0.72rem;
          color: var(--text-tertiary);
          font-family: var(--font-mono);
          letter-spacing: 0.04em;
          padding: 4px 8px;
          border-radius: var(--r-sm);
          transition: color var(--t-fast), background var(--t-fast);
        }
        .top-nav a:hover {
          color: var(--text-primary);
          background: var(--bg-elevated);
        }
`}</style>

      {/* ══════════════════════════════════════
          TOP NAV BAR — quick route access
      ══════════════════════════════════════ */}
      <nav className="top-nav" style={{
        position: "sticky", top: 0, zIndex: 50,
        background: "var(--bg-base)", borderBottom: "1px solid var(--border-faint)",
        padding: "10px 36px",
        display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap",
      }}>
        <span style={{
          fontFamily: "var(--font-mono)", fontSize: "0.65rem",
          color: "var(--text-dim)", letterSpacing: "0.1em", marginRight: "12px",
        }}>ML REF</span>

        {TOOLS.map(t => (
          <Link key={t.route} to={t.route} style={{ color: t.accent }}>
            {t.icon} {t.title.split(" ")[0]}…
          </Link>
        ))}
      </nav>

      {/* ══════════════════════════════════════
          HERO
      ══════════════════════════════════════ */}
      <header style={{
        padding: "64px 36px 52px",
        borderBottom: "1px solid var(--border-faint)",
        position: "relative", overflow: "hidden",
      }}>
        {/* Subtle dot-grid background texture */}
        <div style={{
          position: "absolute", inset: 0,
          backgroundImage: "radial-gradient(circle, var(--border-faint) 1px, transparent 1px)",
          backgroundSize: "32px 32px",
          opacity: 0.7, pointerEvents: "none",
        }} />

        <div style={{ position: "relative", maxWidth: "820px", animation: "fadeIn 0.7s ease both" }}>

          {/* Eyebrow */}
          <div style={{ display: "flex", alignItems: "center", gap: "14px", marginBottom: "22px" }}>
            <div style={{ width: "32px", height: "1px", background: "var(--border-strong)" }} />
            <span style={{
              fontFamily: "var(--font-mono)", fontSize: "0.62rem",
              color: "var(--text-tertiary)", letterSpacing: "0.2em", textTransform: "uppercase",
            }}>Machine Learning · Reference Library · v2.0</span>
          </div>

          {/* Headline */}
          <h1 style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(2.6rem, 5.5vw, 4.2rem)",
            fontWeight: 400, lineHeight: 1.08,
            letterSpacing: "-0.025em",
            color: "var(--text-primary)",
            marginBottom: "22px",
          }}>
            The complete practitioner's<br />
            <span style={{ fontStyle: "italic", fontWeight: 300, color: "var(--text-secondary)" }}>
              reference for every stage
            </span>
          </h1>

          {/* Sub */}
          <p style={{
            fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.75,
            maxWidth: "560px", marginBottom: "40px",
          }}>
            Seven interconnected reference tables covering the complete ML lifecycle — from first
            EDA to deployed, monitored production models. Each table cross-references the others.
            Follow the Playbook to know when to consult each one.
          </p>

          {/* Stat bar */}
          <div style={{
            display: "flex", gap: "0", flexWrap: "wrap",
            borderTop: "1px solid var(--border-subtle)",
            borderBottom: "1px solid var(--border-subtle)",
            padding: "20px 0",
          }}>
            {STATS.map((s, i) => (
              <div key={s.label} style={{
                flex: "1 1 80px", minWidth: "80px",
                paddingRight: "28px",
                borderRight: i < STATS.length - 1 ? "1px solid var(--border-subtle)" : "none",
                marginRight: i < STATS.length - 1 ? "28px" : "0",
              }}>
                <div style={{
                  fontFamily: "var(--font-display)", fontSize: "2.1rem",
                  fontWeight: 400, color: "var(--text-primary)", lineHeight: 1,
                }}>{s.n}</div>
                <div style={{
                  fontFamily: "var(--font-mono)", fontSize: "0.6rem",
                  color: "var(--text-tertiary)", letterSpacing: "0.08em", marginTop: "6px",
                }}>{s.label}</div>
              </div>
            ))}
          </div>

          {/* Primary CTA */}
          <div style={{ marginTop: "32px", display: "flex", gap: "12px", flexWrap: "wrap" }}>
            <Link to="/playbook" style={{
              display: "inline-flex", alignItems: "center", gap: "8px",
              padding: "10px 22px", borderRadius: "var(--r-md)",
              background: "var(--fill-amber)", border: `1px solid var(--accent-amber)`,
              color: "var(--accent-amber)", fontSize: "0.8rem", fontWeight: 600,
              letterSpacing: "0.02em", transition: "all var(--t-fast)",
            }}>
              🧭 Start with the Playbook →
            </Link>
            <Link to="/eda" style={{
              display: "inline-flex", alignItems: "center", gap: "8px",
              padding: "10px 22px", borderRadius: "var(--r-md)",
              background: "transparent", border: "1px solid var(--border-strong)",
              color: "var(--text-tertiary)", fontSize: "0.8rem",
              transition: "all var(--t-fast)",
            }}>
              Jump to EDA →
            </Link>
          </div>
        </div>
      </header>

      {/* ══════════════════════════════════════
          TOOL GRID
      ══════════════════════════════════════ */}
      <section style={{ padding: "44px 36px 0" }}>
        <div style={{ marginBottom: "20px", display: "flex", alignItems: "baseline", gap: "12px" }}>
          <span style={{
            fontFamily: "var(--font-mono)", fontSize: "0.62rem",
            color: "var(--text-dim)", letterSpacing: "0.14em", textTransform: "uppercase",
          }}>Reference Tables</span>
          <span style={{ height: "1px", flex: 1, background: "var(--border-faint)", display: "block" }} />
        </div>
        <div className="home-grid">
          {TOOLS.map((tool, i) => (
            <ToolCard key={tool.route} tool={tool} animDelay={0.05 + i * 0.06} />
          ))}
        </div>
      </section>

      {/* ══════════════════════════════════════
          WORKFLOW SPINE
      ══════════════════════════════════════ */}
      <section style={{ padding: "52px 36px 0" }}>
        <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "44px" }}>
          <div style={{ marginBottom: "10px", display: "flex", alignItems: "baseline", gap: "12px" }}>
            <span style={{
              fontFamily: "var(--font-mono)", fontSize: "0.62rem",
              color: "var(--text-dim)", letterSpacing: "0.14em", textTransform: "uppercase",
            }}>How the tables connect</span>
            <span style={{ height: "1px", flex: 1, background: "var(--border-faint)", display: "block" }} />
          </div>
          <p style={{
            fontSize: "0.8rem", color: "var(--text-tertiary)", lineHeight: 1.65,
            maxWidth: "520px", marginBottom: "28px",
          }}>
            Each table is a layer in a single workflow. The Playbook tells you when to consult
            each one. Follow the arrows left-to-right on a new project.
          </p>
          <div className="flow-row">
            {FLOW.map((step, i) => (
              <FlowNode key={step.label} step={step} index={i} isLast={i === FLOW.length - 1} />
            ))}
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════
          QUICK DECISION GUIDE
      ══════════════════════════════════════ */}
      <section style={{ padding: "52px 36px 0" }}>
        <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "44px" }}>
          <div style={{ marginBottom: "10px", display: "flex", alignItems: "baseline", gap: "12px" }}>
            <span style={{
              fontFamily: "var(--font-mono)", fontSize: "0.62rem",
              color: "var(--text-dim)", letterSpacing: "0.14em", textTransform: "uppercase",
            }}>Quick decision guide</span>
            <span style={{ height: "1px", flex: 1, background: "var(--border-faint)", display: "block" }} />
          </div>
          <p style={{
            fontSize: "0.8rem", color: "var(--text-tertiary)", lineHeight: 1.65,
            maxWidth: "480px", marginBottom: "20px",
          }}>
            Navigate by situation — click any card to jump directly to the relevant section.
          </p>
          <div className="decision-grid">
            {DECISIONS.map(({ q, a, r, accent }) => (
              <Link key={q} to={r} style={{
                display: "block", padding: "14px 18px",
                background: "var(--bg-surface)",
                transition: "background var(--t-fast)",
              }}
                onMouseEnter={e => e.currentTarget.style.background = "var(--bg-elevated)"}
                onMouseLeave={e => e.currentTarget.style.background = "var(--bg-surface)"}
              >
                <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: "5px" }}>
                  {q}
                </div>
                <div style={{ fontSize: "0.72rem", color: accent, fontFamily: "var(--font-mono)" }}>
                  {a}
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>
      {/* ══════════════════════════════════════
          PHILOSOPHY FOOTER
      ══════════════════════════════════════ */}
      <footer style={{
        padding: "64px 36px 52px",
        marginTop: "64px",
        borderTop: "1px solid var(--border-subtle)",
      }}>
        <div style={{ maxWidth: "640px", margin: "0 auto", textAlign: "center" }}>
          <blockquote style={{
            fontFamily: "var(--font-display)", fontStyle: "italic",
            fontSize: "1.42rem", fontWeight: 300,
            color: "var(--text-secondary)", lineHeight: 1.55,
            marginBottom: "22px",
          }}>
            "Every leakage, every wrong CV strategy, every over-tuned threshold is a
            violation of the same principle — information from a higher trust level
            contaminating a lower one."
          </blockquote>
          <div style={{
            fontFamily: "var(--font-mono)", fontSize: "0.6rem",
            color: "var(--text-dim)", letterSpacing: "0.12em",
            textTransform: "uppercase", marginBottom: "36px",
          }}>
            The Hierarchy of Trust · Training &lt; CV Score &lt; Held-out Test &lt; Production
          </div>

          {/* Tool index footer links */}
          <div style={{
            display: "flex", justifyContent: "center",
            gap: "6px", flexWrap: "wrap", marginBottom: "32px",
          }}>
            {TOOLS.map(t => (
              <Link key={t.route} to={t.route} style={{
                padding: "4px 12px", borderRadius: "var(--r-sm)",
                fontSize: "0.68rem", fontFamily: "var(--font-mono)",
                background: t.fill, color: t.accent,
                border: `1px solid ${t.accent}30`,
                transition: "opacity var(--t-fast)",
              }}>
                {t.icon} {t.index} {t.title.split(" ").slice(0, 2).join(" ")}
              </Link>
            ))}
          </div>

          <div style={{
            display: "flex", justifyContent: "space-between",
            flexWrap: "wrap", gap: "10px",
            paddingTop: "24px", borderTop: "1px solid var(--border-faint)",
          }}>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.58rem", color: "var(--text-dim)" }}>
              sklearn · scipy · PyTorch · Skorch · MLflow · FastAPI · Docker · Evidently · and 40+ more libraries
            </span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.58rem", color: "var(--text-dim)" }}>
              ML Reference Library · 2026
            </span>
          </div>
        </div>
      </footer>

    </div>
  );
}