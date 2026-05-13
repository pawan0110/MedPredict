import Groq from "groq-sdk";
import { tavily } from "@tavily/core";
import readline from "node:readline/promises";
import "dotenv/config";

const tvly = tavily({ apikey: process.env.TAVILY_API_KEY });
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const messages = [
    {
      role: "system",
      content: `You are a helpful assistant. Use tools when real-time data is required.
      current date and time: ${new Date().zatoUTCString()}`,
    },
  ];

  while (true) {
    const question = await rl.question("You: ");

    if (question.toLowerCase() === "bye") {
      console.log("Assistant: Goodbye!");
      break;
    }

    messages.push({ role: "user", content: question });

    // 1️⃣ First model call (tools allowed)
    const completion = await groq.chat.completions.create({
      model: "openai/gpt-oss-20b",
      messages,
      tools: [
        {
          type: "function",
          function: {
            name: "webSearch",
            description: "Search real-time information on the internet",
            parameters: {
              type: "object",
              properties: {
                query: { type: "string" },
              },
              required: ["query"],
            },
          },
        },
      ],
      tool_choice: "auto",
    });

    const assistantMessage = completion.choices[0].message;

    // 2️⃣ No tool required
    if (!assistantMessage.tool_calls) {
      console.log("Assistant:", assistantMessage.content);
      messages.push(assistantMessage);
      continue;
    }

    // 3️⃣ Handle tool calls
    for (const toolCall of assistantMessage.tool_calls) {
      if (toolCall.function.name === "webSearch") {
        const args = JSON.parse(toolCall.function.arguments);
        const toolResult = await webSearch(args);

        messages.push(assistantMessage);
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: toolResult,
        });

        // 4️⃣ Final model call (tools disabled)
        const finalCompletion = await groq.chat.completions.create({
          model: "openai/gpt-oss-20b",
          messages,
          tool_choice: "none",
        });

        const finalMessage = finalCompletion.choices[0].message;
        console.log("Assistant:", finalMessage.content);
        messages.push(finalMessage);
      }
    }
  }

  rl.close();
}

async function webSearch({ query }) {
  console.log("🔍 Searching:", query);

  const response = await tvly.search(query);

  if (!response.results || response.results.length === 0) {
    return "No reliable information was found.";
  }

  const summary = response.results
    .slice(0, 3)
    .map(
      (r, i) => `${i + 1}. ${r.content} (Source: ${r.url})`
    )
    .join("\n\n");

  return `Search Results:\n\n${summary}`;
}

main();
