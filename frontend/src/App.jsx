import { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import ChatMessage from "./components/ChatMessage.jsx";
import ChatInput from "./components/ChatInput.jsx";
import ChartCard from "./components/ChartCard.jsx";

export default function App() {
  // Hardcoded demo messages
  const messages = [
    {
      id: 1,
      role: "assistant",
      type: "text",
      content:
        "Hey! I’m AnalyiXpert. Ask me anything NBA — players, teams, advanced stats, projections, and more.",
    },
    {
      id: 2,
      role: "user",
      type: "text",
      content:
        "Who were the top PPG leaders last season? Can you show a quick chart?",
    },
    {
      id: 3,
      role: "assistant",
      type: "chart",
      content:
        "Here’s a snapshot of PPG for a few stars. I can break this down by month, game log, or add error bands.",
    },
  ];

  // Lift the input state so sidebar clicks can populate the input box
  const [inputValue, setInputValue] = useState("");

  const handleSampleClick = (text) => {
    setInputValue(text);
  };

  const handleSend = (text) => {
    // For now we just clear. Wire this to your backend later.
    setInputValue("");
    // You could also append a new user message to `messages` here in the future.
  };

  return (
    <div className="app">
      <Sidebar onSampleClick={handleSampleClick} />

      <main className="chat">
        <header className="chat__header">
          <div className="chat__title">Chat</div>
          <div className="chat__sub">NBA insights powered by your data</div>
        </header>

        <section className="chat__messages">
          {messages.map((m) => {
            if (m.type === "chart") {
              return (
                <ChatMessage key={m.id} role={m.role}>
                  <p className="msg__text">{m.content}</p>
                  <ChartCard />
                </ChatMessage>
              );
            }
            return (
              <ChatMessage key={m.id} role={m.role}>
                <p className="msg__text">{m.content}</p>
              </ChatMessage>
            );
          })}
        </section>

        <ChatInput
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSend}
        />
      </main>
    </div>
  );
}
