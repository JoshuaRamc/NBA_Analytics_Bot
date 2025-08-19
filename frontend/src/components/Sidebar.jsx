export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand__logo">🏀</div>
        <div className="brand__name">
          NBA <span>AnalyiXpert</span>
        </div>
      </div>

      {/* Sample questions removed as requested */}

      <div className="panel panel--foot">
        <div className="hint">
          Pro tip: Ask for charts — I can visualize time series, rankings, and shooting
          splits using your data.
        </div>
      </div>
    </aside>
  );
}
