export default function Sidebar({ onSampleClick = () => {} }) {
  // Sample questions tailored to your dataset columns (PTS, TRB, AST, 3P%, eFG%, FGA, Awards, etc.)
  const samples = [
    "Show top 10 PTS leaders in 2024–25 with Team and Pos.",
    "Compare Shai Gilgeous-Alexander, Giannis Antetokounmpo, Nikola Jokić — PTS, TRB, AST.",
    "Who has the highest eFG% among players with ≥ 15 FGA per game?",
    "List players with 3P% ≥ 40% on ≥ 6 3PA per game.",
    "Find all players with ≥ 25 PTS, ≥ 6 AST, ≥ 5 TRB per game.",
  ];

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand__logo">🏀</div>
        <div className="brand__name">
          NBA <span>AnalyiXpert</span>
        </div>
      </div>

      <div className="panel">
        <div className="panel__title">Sample questions</div>
        <ul className="sample-list">
          {samples.map((q, i) => (
            <li
              key={i}
              className="sample-list__item"
              role="button"
              tabIndex={0}
              onClick={() => onSampleClick(q)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") onSampleClick(q);
              }}
              title="Click to use this prompt"
            >
              <span className="dot" />
              {q}
            </li>
          ))}
        </ul>
      </div>

      <div className="panel panel--foot">
        <div className="hint">
          Pro tip: Ask for charts — I can visualize time series, rankings, and
          shooting splits using your data.
        </div>
      </div>
    </aside>
  );
}
