import { useEffect, useRef, useState } from "react";
import type { StatusPayload } from "@/lib/types";
import { getMockStatus } from "@/lib/mock-status";

const ENDPOINT = "http://localhost:8000/status";
const POLL_MS = 400;

export function useStatus() {
  const [data, setData] = useState<StatusPayload>(() => getMockStatus());
  const [connected, setConnected] = useState(false);
  const failedRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 800);
        const res = await fetch(ENDPOINT, { signal: ctrl.signal });
        clearTimeout(t);
        if (!res.ok) throw new Error("bad status");
        const json = (await res.json()) as Partial<StatusPayload>;
        if (cancelled) return;
        failedRef.current = 0;
        setConnected(true);
        // Merge defensively — backend may not provide every field
        setData((prev) => ({
          frame: json.frame ?? prev.frame,
          status: json.status ?? prev.status,
          alerts: json.alerts ?? prev.alerts,
          logs: json.logs ?? prev.logs,
          metrics: { ...prev.metrics, ...(json.metrics ?? {}) },
        }));
      } catch {
        failedRef.current++;
        if (failedRef.current > 2) {
          setConnected(false);
          // Fallback to mock so UI stays alive during dev
          setData(getMockStatus());
        }
      } finally {
        if (!cancelled) timer = setTimeout(tick, POLL_MS);
      }
    };

    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  return { data, connected };
}
