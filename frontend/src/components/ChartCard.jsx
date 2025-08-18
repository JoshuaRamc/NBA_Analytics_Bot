import {
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

const data = [
  { player: "SGA", ppg: 32.7 },
  { player: "Giannis", ppg: 30.4 },
  { player: "Jokić", ppg: 29.6 },
  { player: "Luka", ppg: 28.2 },
  { player: "Brunson", ppg: 26.0 },
];

export default function ChartCard() {
  return (
    <div className="chart-card">
      <div className="chart-card__title">PPG Leaders (demo data)</div>
      <div className="chart-card__wrap">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eaf6f6" />
            <XAxis dataKey="player" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="ppg" radius={[6, 6, 0, 0]} fill="#9ce6e6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
