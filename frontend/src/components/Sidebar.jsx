// Sidebar.jsx
export default function Sidebar({ onSelectQuestion }) {
  const sampleQuestions = [
    "Show me top 5 scorers this season",
    "Who has the highest 3-point percentage this year?",
    "Which player has the most rebounds this season?",
    "Who leads the league in assists?",
    "Compare LeBron James and Stephen Curry stats this season",
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
        <h3 className="panel__title">Try asking:</h3>
        <ul className="sample-list">
          {sampleQuestions.map((q, i) => (
            <li key={i} className="sample-list__item">
              <button
                type="button"
                className="sample-btn"
                onClick={() => onSelectQuestion?.(q)}
                aria-label={`Use sample question: ${q}`}
              >
                <span className="sample-btn__icon" aria-hidden>💡</span>
                <span className="sample-btn__text">{q}</span>
                <span className="sample-btn__chev" aria-hidden>›</span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="panel panel--foot">
        <div className="hint">
          Pro tip: Ask for charts — I can visualize time series, rankings, and shooting
          splits using your data.
        </div>
      </div>
    </aside>
  );
}
