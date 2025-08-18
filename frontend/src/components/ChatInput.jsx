import { useCallback } from "react";

export default function ChatInput({ value, onChange, onSend }) {
  const handleSend = useCallback(
    (e) => {
      e.preventDefault();
      const text = (value || "").trim();
      if (!text) return;
      onSend?.(text);
    },
    [value, onSend]
  );

  return (
    <form className="chat-input" onSubmit={handleSend}>
      <input
        className="chat-input__field"
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        placeholder="Ask about players, teams, or advanced metrics…"
      />
      <button className="chat-input__btn" type="submit" aria-label="Send">
        Send
      </button>
    </form>
  );
}
