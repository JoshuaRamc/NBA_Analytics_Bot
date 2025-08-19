import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ChartRenderer from "./ChartRenderer.jsx";

/* Chat bubble that supports Markdown and optional chart block */
export default function ChatMessage({ role = "assistant", content = "", type = "text", chart }) {
  const isUser = role === "user";

  return (
    <div className={`msg ${isUser ? "msg--user" : "msg--assistant"}`}>
      <div
        className={`msg__avatar ${isUser ? "msg__avatar--user" : "msg__avatar--assistant"}`}
      >
        {isUser ? "🧑" : "🤖"}
      </div>

      <div className="msg__bubble">
        {/* Markdown message */}
        {content ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        ) : null}

        {/* Optional chart, only render if type === 'chart' and spec present */}
        {type === "chart" && chart ? (
          <div className="chart-card" style={{ marginTop: 8 }}>
            {chart.title ? (
              <div className="chart-card__title">
                {chart.title} {chart.subtitle ? `— ${chart.subtitle}` : ""}
              </div>
            ) : null}
            <div className="chart-card__wrap">
              <ChartRenderer spec={chart} />
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
