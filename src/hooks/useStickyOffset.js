import { useRef, useState, useEffect } from "react";

export function useStickyOffset(fallbackPx = 80) {
  const ref = useRef(null);             // this ref gets attached to the filter bar div
  const [height, setHeight] = useState(fallbackPx);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new ResizeObserver(entries => {
      setHeight(Math.ceil(entries[0].contentRect.height));
    });

    observer.observe(el);
    return () => observer.disconnect();
  }, []);  // empty deps — runs once on mount, observer handles updates

  return [ref, height];
}