import { NavLink } from "react-router-dom";

// Every route in the app, in the order we want them displayed in the nav.
// Adding a new table later? Just add one entry here — no other file needs to change.
const NAV_ITEMS = [
  { route: "/",         label: "Home",       icon: "⌂",  accent: "#8a7248" },
  { route: "/playbook", label: "Playbook",   icon: "🧭", accent: "#8a7248" },
  { route: "/eda",      label: "EDA",        icon: "🔭", accent: "#2e7268" },
  { route: "/features", label: "Features",   icon: "⚗️", accent: "#3e6e52" },
  { route: "/sklearn",  label: "sklearn",    icon: "📐", accent: "#3a5e7a" },
  { route: "/problems", label: "Problems",   icon: "🩺", accent: "#784054" },
  { route: "/torch",    label: "PyTorch",    icon: "🔥", accent: "#5a4880" },
  { route: "/deploy",   label: "MLOps",      icon: "🚀", accent: "#2e6660" },
];

export default function Header() {
  return (
    <header style={{
      position:     "sticky",
      top:          0,
      zIndex:       100,                         // Sits above sticky table headers (z-index 5-10)
      background:   "rgba(14,13,12,0.93)",                 // Semi-transparent so content scrolls behind it
      backdropFilter: "blur(12px)",              // Frosted-glass effect on modern browsers
      borderBottom: "1px solid var(--border-faint)",
      padding:      "0 20px",
      display:      "flex",
      alignItems:   "center",
      gap:          "4px",
      height:       "44px",
      flexWrap:     "nowrap",
      overflowX:    "auto",                      // Scrollable on narrow screens
    }}>

      {/* Library wordmark — always links back to the homepage */}
      <NavLink to="/" style={{ marginRight: "12px", flexShrink: 0 }}>
        <span style={{
          fontFamily:     "var(--font-mono)",
          fontSize:       "0.62rem",
          fontWeight:     600,
          color:          "var(--text-dim)",
          letterSpacing:  "0.14em",
          textTransform:  "uppercase",
        }}>ML·REF</span>
      </NavLink>

      {/* Thin vertical divider between wordmark and nav links */}
      <div style={{
        width:      "1px",
        height:     "18px",
        background: "var(--border-default)",
        flexShrink: 0,
        marginRight: "8px",
      }} />

      {/* Navigation links */}
      {NAV_ITEMS.map(({ route, label, icon, accent }) => (
        <NavLink
          key={route}
          to={route}
          // React Router's NavLink passes { isActive } to the style function,
          // so we can change appearance based on whether this IS the current page.
          style={({ isActive }) => ({
            display:       "flex",
            alignItems:    "center",
            gap:           "5px",
            padding:       "4px 10px",
            borderRadius:  "5px",
            fontSize:      "0.72rem",
            fontFamily:    "'DM Sans', sans-serif",
            fontWeight:    isActive ? 600 : 400,
            whiteSpace:    "nowrap",
            flexShrink:    0,
            textDecoration: "none",
            transition:    "background 0.15s, color 0.15s",

            // Active page gets a tinted background and the route's accent colour.
            // Inactive pages are dim until hovered — but hover is handled via CSS below.
            background: isActive ? `${accent}1a` : "transparent",
            color:      isActive ? accent        : "var(--text-tertiary)",
            border:     isActive ? `1px solid ${accent}33` : "1px solid transparent",
          })}
          // end prop is crucial for the "/" home route — without it, the Home link
          // would appear "active" on EVERY page because every path starts with "/".
          end={route === "/"}
        >
          <span style={{ fontSize: "0.78rem" }}>{icon}</span>
          {label}
        </NavLink>
      ))}
    </header>
  );
}