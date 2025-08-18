export default function ChatMessage({ role = "assistant", children }) {
  const isUser = role === "user";
  return (
    <div className={`msg ${isUser ? "msg--user" : "msg--assistant"}`}>
      <div className={`msg__avatar ${isUser ? "msg__avatar--user" : "msg__avatar--assistant"}`}>
        {isUser ? "🧑" : "🤖"}
      </div>
      <div className="msg__bubble">
        {children}
      </div>
    </div>
  );
}
