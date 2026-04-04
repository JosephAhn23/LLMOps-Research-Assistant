/**
 * React + TypeScript monitoring dashboard.
 * Covers: TypeScript explicit, data visualization, real-time WebSocket
 */
import { useState, useEffect, useCallback, useRef } from "react";

interface PipelineMetrics {
  latency_p50: number;
  latency_p90: number;
  latency_p99: number;
  tokens_per_second: number;
  error_rate: number;
  cache_hit_rate: number;
  active_connections: number;
  queries_per_minute: number;
}

interface RAGASMetrics {
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number;
  timestamp: number;
}

interface StreamMessage {
  type: "token" | "done" | "error" | "status" | "progress";
  content?: string;
  full_response?: string;
  latency_ms?: number;
  tokens_per_second?: number;
  message?: string;
}

interface ExperimentResult {
  name: string;
  p_value: number;
  relative_lift: number;
  is_significant: boolean;
  winner: string | null;
}

function useWebSocketStream(url: string) {
  const ws = useRef<WebSocket | null>(null);
  const [response, setResponse] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [metadata, setMetadata] = useState<Partial<StreamMessage>>({});
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => setConnected(true);
    ws.current.onclose = () => setConnected(false);

    ws.current.onmessage = (event: MessageEvent) => {
      const msg: StreamMessage = JSON.parse(event.data);

      if (msg.type === "token" && msg.content) {
        setResponse((prev) => prev + msg.content);
      } else if (msg.type === "done") {
        setIsStreaming(false);
        setMetadata(msg);
      } else if (msg.type === "error") {
        setIsStreaming(false);
        console.error("Stream error:", msg.message);
      }
    };

    return () => ws.current?.close();
  }, [url]);

  const sendQuery = useCallback((query: string) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      setResponse("");
      setIsStreaming(true);
      ws.current.send(JSON.stringify({ query }));
    }
  }, []);

  return { response, isStreaming, metadata, connected, sendQuery };
}

function MetricCard({
  label,
  value,
  unit,
  trend,
  color = "#4f8ef7",
}: {
  label: string;
  value: number | string;
  unit?: string;
  trend?: "up" | "down" | "neutral";
  color?: string;
}) {
  const trendIcon = trend === "up" ? "↑" : trend === "down" ? "↓" : "→";
  const trendColor = trend === "up" ? "#4fce7f" : trend === "down" ? "#f74f4f" : "#aaa";

  return (
    <div
      style={{
        background: "#1a1a2e",
        border: "1px solid #2a2a4a",
        borderRadius: 8,
        padding: "16px 20px",
        minWidth: 160,
      }}
    >
      <div style={{ color: "#888", fontSize: 12, marginBottom: 6 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
        <span style={{ color, fontSize: 28, fontWeight: 700 }}>{value}</span>
        {unit && <span style={{ color: "#666", fontSize: 13 }}>{unit}</span>}
        {trend && <span style={{ color: trendColor, fontSize: 14 }}>{trendIcon}</span>}
      </div>
    </div>
  );
}

function RAGASChart({ history }: { history: RAGASMetrics[] }) {
  const metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] as const;
  const colors = ["#4f8ef7", "#4fce7f", "#f7c94f", "#f74f9a"];

  return (
    <div style={{ background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 8, padding: 20 }}>
      <div style={{ color: "#ddd", fontSize: 14, marginBottom: 16, fontWeight: 600 }}>RAGAS Metrics</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {metrics.map((metric, i) => {
          const latest = history[history.length - 1]?.[metric] ?? 0;
          const prev = history[history.length - 2]?.[metric] ?? latest;
          const trend = latest > prev ? "up" : latest < prev ? "down" : "neutral";
          return (
            <MetricCard
              key={metric}
              label={metric.replace(/_/g, " ")}
              value={latest.toFixed(3)}
              trend={trend}
              color={colors[i]}
            />
          );
        })}
      </div>
    </div>
  );
}

function StreamingQueryPanel() {
  const { response, isStreaming, metadata, connected, sendQuery } = useWebSocketStream("ws://localhost:8000/ws/query");
  const [query, setQuery] = useState("");

  return (
    <div style={{ background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 8, padding: 20 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
        <div style={{ color: "#ddd", fontSize: 14, fontWeight: 600 }}>Live Query</div>
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: connected ? "#4fce7f" : "#f74f4f",
          }}
        />
        <span style={{ color: "#666", fontSize: 12 }}>{connected ? "connected" : "disconnected"}</span>
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendQuery(query)}
          placeholder="Ask a question..."
          style={{
            flex: 1,
            background: "#0d0d1a",
            border: "1px solid #2a2a4a",
            borderRadius: 6,
            padding: "8px 12px",
            color: "#ddd",
            fontSize: 14,
            outline: "none",
          }}
        />
        <button
          onClick={() => sendQuery(query)}
          disabled={isStreaming || !connected}
          style={{
            background: isStreaming ? "#333" : "#4f8ef7",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            padding: "8px 16px",
            cursor: isStreaming ? "not-allowed" : "pointer",
            fontSize: 14,
          }}
        >
          {isStreaming ? "streaming..." : "send"}
        </button>
      </div>

      <div
        style={{
          background: "#0d0d1a",
          borderRadius: 6,
          padding: 12,
          minHeight: 120,
          maxHeight: 300,
          overflowY: "auto",
          color: "#ccc",
          fontSize: 14,
          lineHeight: 1.6,
          fontFamily: "monospace",
        }}
      >
        {response || <span style={{ color: "#444" }}>response appears here...</span>}
      </div>

      {metadata.latency_ms && (
        <div style={{ display: "flex", gap: 16, marginTop: 12 }}>
          <span style={{ color: "#666", fontSize: 12 }}>Latency: {metadata.latency_ms}ms</span>
          <span style={{ color: "#666", fontSize: 12 }}>Rate: {metadata.tokens_per_second} tok/s</span>
        </div>
      )}
    </div>
  );
}

function ExperimentPanel({ experiments }: { experiments: ExperimentResult[] }) {
  return (
    <div style={{ background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 8, padding: 20 }}>
      <div style={{ color: "#ddd", fontSize: 14, fontWeight: 600, marginBottom: 16 }}>A/B Experiments</div>
      {experiments.length === 0 ? (
        <div style={{ color: "#444", fontSize: 13 }}>No experiments running</div>
      ) : (
        experiments.map((exp, i) => (
          <div
            key={i}
            style={{
              padding: "10px 0",
              borderBottom: i < experiments.length - 1 ? "1px solid #2a2a4a" : "none",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ color: "#ccc", fontSize: 13 }}>{exp.name}</span>
              <span
                style={{
                  fontSize: 11,
                  padding: "2px 8px",
                  borderRadius: 4,
                  background: exp.is_significant ? "#1a3a1a" : "#2a2a1a",
                  color: exp.is_significant ? "#4fce7f" : "#f7c94f",
                }}
              >
                {exp.is_significant ? "significant" : "collecting"}
              </span>
            </div>
            <div style={{ display: "flex", gap: 16, marginTop: 6 }}>
              <span style={{ color: "#666", fontSize: 12 }}>p={exp.p_value.toFixed(4)}</span>
              <span style={{ color: exp.relative_lift > 0 ? "#4fce7f" : "#f74f4f", fontSize: 12 }}>
                {exp.relative_lift > 0 ? "+" : ""}
                {(exp.relative_lift * 100).toFixed(1)}% lift
              </span>
              {exp.winner && <span style={{ color: "#4f8ef7", fontSize: 12 }}>winner: {exp.winner}</span>}
            </div>
          </div>
        ))
      )}
    </div>
  );
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<PipelineMetrics>({
    latency_p50: 0,
    latency_p90: 0,
    latency_p99: 0,
    tokens_per_second: 0,
    error_rate: 0,
    cache_hit_rate: 0,
    active_connections: 0,
    queries_per_minute: 0,
  });
  const [ragasHistory, setRagasHistory] = useState<RAGASMetrics[]>([]);
  const [experiments, setExperiments] = useState<ExperimentResult[]>([]);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const [metricsRes, ragasRes, expRes] = await Promise.all([
          fetch("/api/metrics"),
          fetch("/api/ragas/history"),
          fetch("/api/experiments"),
        ]);
        if (metricsRes.ok) setMetrics(await metricsRes.json());
        if (ragasRes.ok) setRagasHistory(await ragasRes.json());
        if (expRes.ok) setExperiments(await expRes.json());
      } catch (e) {
        console.error("Failed to fetch metrics:", e);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div
      style={{
        background: "#0d0d1a",
        minHeight: "100vh",
        padding: 24,
        fontFamily: "system-ui, sans-serif",
        color: "#ddd",
      }}
    >
      <div style={{ maxWidth: 1400, margin: "0 auto" }}>
        <h1 style={{ color: "#fff", fontSize: 20, fontWeight: 700, marginBottom: 24 }}>
          LLMOps Research Assistant - Dashboard
        </h1>

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 24 }}>
          <MetricCard label="p50 Latency" value={metrics.latency_p50} unit="ms" />
          <MetricCard label="p99 Latency" value={metrics.latency_p99} unit="ms" />
          <MetricCard label="Tokens/sec" value={metrics.tokens_per_second} color="#4fce7f" />
          <MetricCard
            label="Error Rate"
            value={(metrics.error_rate * 100).toFixed(2)}
            unit="%"
            color="#f74f4f"
          />
          <MetricCard
            label="Cache Hit Rate"
            value={(metrics.cache_hit_rate * 100).toFixed(1)}
            unit="%"
            color="#f7c94f"
          />
          <MetricCard label="Active Connections" value={metrics.active_connections} color="#f74f9a" />
          <MetricCard label="Queries/min" value={metrics.queries_per_minute} />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
          <RAGASChart history={ragasHistory} />
          <ExperimentPanel experiments={experiments} />
        </div>

        <StreamingQueryPanel />
      </div>
    </div>
  );
}
