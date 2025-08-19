import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

/**
 * Renders a Recharts chart from a backend chart spec.
 * Expected spec:
 * {
 *   chartType: "bar" | "line",
 *   xKey: string,
 *   series: [{ dataKey: string, label?: string }, ...],
 *   data: Array<Record<string, any>>,
 *   xLabel?: string,
 *   yLabel?: string
 * }
 */
export default function ChartRenderer({ spec }) {
  if (!spec || !spec.data || !Array.isArray(spec.data) || spec.data.length === 0) {
    return <div style={{ padding: "8px 12px", color: "#5a6b73" }}>No chartable data.</div>;
  }

  const { chartType, xKey, series = [], data } = spec;

  const renderCommon = (children) => (
    <>
      <CartesianGrid strokeDasharray="3 3" vertical={false} />
      <XAxis dataKey={xKey} />
      <YAxis />
      <Tooltip />
      {series.length > 1 ? <Legend /> : null}
      {children}
    </>
  );

  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer width="100%" height="100%">
        {chartType === "line" ? (
          <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            {renderCommon(
              series.map((s, idx) => (
                <Line
                  key={idx}
                  type="monotone"
                  dataKey={s.dataKey}
                  name={s.label || s.dataKey}
                  dot={false}
                  strokeWidth={2}
                />
              ))
            )}
          </LineChart>
        ) : (
          <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            {renderCommon(
              series.map((s, idx) => (
                <Bar
                  key={idx}
                  dataKey={s.dataKey}
                  name={s.label || s.dataKey}
                  radius={[6, 6, 0, 0]}
                  // Fill intentionally not specified to follow your global styling
                />
              ))
            )}
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
