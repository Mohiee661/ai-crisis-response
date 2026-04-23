export type SystemStatus = "SAFE" | "MONITOR" | "ALERT";
export type Severity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type AlertType = "FIRE" | "FALL" | "INTRUSION" | "SMOKE" | "WEAPON" | "CROWD";

export interface AlertItem {
  id: string;
  type: AlertType;
  severity: Severity;
  cameraId: string;
  timestamp: string;
  message?: string;
}

export interface LogItem {
  id: string;
  timestamp: string;
  level: "INFO" | "WARN" | "ERROR" | "AGENT";
  message: string;
}

export interface Metrics {
  confidence: number; // 0-100
  fps: number;
  latencyMs: number;
  modelsActive: number;
  uptime: string;
}

export interface StatusPayload {
  frame: string | null; // base64 (no prefix or with prefix)
  status: SystemStatus;
  alerts: AlertItem[];
  logs: LogItem[];
  metrics: Metrics;
}
