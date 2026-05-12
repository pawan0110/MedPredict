const input = document.querySelector("#input");
const chatContainer = document.querySelector("#chatContainer");
const askBtn = document.querySelector("#askBtn");

const threadId = Date.now().toString(36) + Math.random().toString(36).substring(2,8);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

askBtn?.addEventListener("click", handleSend);


if (!threadId || typeof threadId !== "string") {
  threadId = "default-thread";
}

async function generate(text,threadId) {
  if (!text.trim()) return;

  // User message
  const userMsg = document.createElement("div");
  userMsg.className =
    "my-4 bg-neutral-800 text-white p-3 rounded-xl ml-auto max-w-fit";
  userMsg.textContent = text;
  chatContainer.appendChild(userMsg);
  input.value = "";

  // Loading message (below user message)
  const loading = document.createElement("div");
  loading.className =
    "my-4 text-neutral-400 mr-auto flex items-center gap-1";
  loading.innerHTML = `
    <span class="animate-bounce">.</span>
    <span class="animate-bounce [animation-delay:150ms]">.</span>
    <span class="animate-bounce [animation-delay:300ms]">.</span>
  `;
  chatContainer.appendChild(loading);

  chatContainer.scrollTop = chatContainer.scrollHeight;

  // Assistant response
  const assistantMessage = await callServer(text, threadId);

  loading.remove();

  const assistantMsg = document.createElement("div");
  assistantMsg.className =
    "my-4 bg-neutral-700 text-white p-3 rounded-xl mr-auto max-w-fit";
  assistantMsg.textContent = assistantMessage;
  chatContainer.appendChild(assistantMsg);

  chatContainer.scrollTop = chatContainer.scrollHeight;
}


async function callServer(inputText, threadId) {
  const response = await fetch("http://localhost:8080/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({threadId, message: inputText }),
  });

  if (!response.ok) {
    throw new Error("Error generating the response");
  }

  const result = await response.json(); 
  return result.message;
}

async function handleClick(e) {
  if (e.key === "Enter") {
    const text = input.value.trim();
    if (!text) return;
    await generate(text, threadId);
  }
}

async function handleSend() {
  const text = input.value.trim();
  if (!text) return;
  await generate(text, threadId);
}
