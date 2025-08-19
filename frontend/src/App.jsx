import { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import ChatMessage from "./components/ChatMessage.jsx";
import ChatInput from "./components/ChatInput.jsx";
import { API } from "./api.js";

export default function App() {
  // Seed with just one friendly assistant welcome message
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: "assistant",
      type: "text",
      content:
        "Hey! I’m AnalyiXpert. Ask me anything NBA — players, teams, advanced stats, projections, and more.",
    },
  ]);

  const [inputValue, setInputValue] = useState("");
  const [pending, setPending] = useState(false);

  const handleSend = async (text) => {
    if (!text.trim()) return;

    const userMsg = {
      id: Date.now(),
      role: "user",
      type: "text",
      content: text,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInputValue("");
    setPending(true);

    // placeholder while waiting
    const placeholderId = Date.now() + 1;
    setMessages((prev) => [
      ...prev,
      { id: placeholderId, role: "assistant", type: "text", content: "_Thinking…_" },
    ]);

    try {
      const res = await fetch(API.ask, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || `Request failed with ${res.status}`);
      }

      const data = await res.json();
      const answer = data?.answer ?? "I couldn't generate a response.";

      // Replace placeholder with the LLM markdown answer
      setMessages((prev) =>
        prev.map((m) => (m.id === placeholderId ? { ...m, content: answer } : m))
      );

      // If backend returned a chart spec, append a chart message after the markdown
      if (data?.chart) {
        const chartMsg = {
          id: placeholderId + 1,
          role: "assistant",
          type: "chart",
          content:
            data.chart.title ||
            "Here’s a visualization based on your question.",
          chart: data.chart, // pass full spec through
        };
        setMessages((prev) => [...prev, chartMsg]);
      }
    } catch (e) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === placeholderId
            ? {
                ...m,
                content:
                  `**Error**: could not reach the NBA AnalyiXpert API.\n\n> ${e?.message || ""}`,
              }
            : m
        )
      );
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="app">
      <Sidebar />

      <main className="chat">
        <header className="chat__header">
          <div className="chat__title">Chat</div>
          <div className="chat__sub">
            NBA insights powered by your data {pending ? "• generating…" : ""}
          </div>
        </header>

        {/* scrollable messages area */}
        <section className="chat__messages">
          {messages.map((m) => (
            <ChatMessage
              key={m.id}
              role={m.role}
              content={m.content}
              type={m.type}
              chart={m.chart} // may be undefined
            />
          ))}
        </section>

        {/* input always pinned at bottom */}
        <ChatInput value={inputValue} onChange={setInputValue} onSend={handleSend} />
      </main>
    </div>
  );
}
