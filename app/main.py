from fastapi import FastAPI
from starlette.responses import HTMLResponse
from app.llm import ChatSession
from app.schemas import AskRequest, AskResponse, ClearResponse

app = FastAPI(title="RAG System")

# persistent session to preserve conversation history
session = ChatSession()

@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>RAG Test UI</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; max-width: 800px; }
      #history { border: 1px solid #ddd; padding: 10px; height: 400px; overflow: auto; background: #fafafa; }
      .turn { margin: 6px 0; }
      .user { color: #0b5; }
      .assistant { color: #05f; }
      #controls { margin-top: 10px; display:flex; gap:8px; }
      #query { flex:1; padding:6px; }
      button { padding:6px 10px; }
    </style>
  </head>
  <body>
    <h2>RAG Test UI</h2>
    <div id="history" aria-live="polite"></div>

    <div id="controls">
      <input id="query" placeholder="Type a question..." />
      <button id="send">Send</button>
      <button id="clear">Clear All</button>
    </div>

    <script>
      const historyEl = document.getElementById('history');
      const queryEl = document.getElementById('query');
      const sendBtn = document.getElementById('send');
      const clearBtn = document.getElementById('clear');

      function appendTurn(role, text) {
        const div = document.createElement('div');
        div.className = 'turn ' + (role === 'user' ? 'user' : 'assistant');
        div.textContent = (role === 'user' ? 'You: ' : 'Assistant: ') + text;
        historyEl.appendChild(div);
        historyEl.scrollTop = historyEl.scrollHeight;
      }

      sendBtn.addEventListener('click', async () => {
        const q = queryEl.value.trim();
        if (!q) return;
        appendTurn('user', q);
        queryEl.value = '';
        sendBtn.disabled = true;
        try {
          const res = await fetch('/ask', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query: q })
          });
          if (!res.ok) {
            appendTurn('assistant', 'Error: ' + res.statusText);
          } else {
            const data = await res.json();
            appendTurn('assistant', data.answer || '(no answer)');
          }
        } catch (err) {
          appendTurn('assistant', 'Network error');
        } finally {
          sendBtn.disabled = false;
        }
      });

      clearBtn.addEventListener('click', async () => {
        clearBtn.disabled = true;
        try {
          const res = await fetch('/clear', { method: 'POST' });
          if (!res.ok) {
            appendTurn('assistant', 'Error clearing history: ' + res.statusText);
          } else {
            historyEl.innerHTML = '';
            appendTurn('assistant', 'Conversation history cleared.');
          }
        } catch (err) {
          appendTurn('assistant', 'Network error while clearing history');
        } finally {
          clearBtn.disabled = false;
        }
      });

      // allow Enter to send
      queryEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          sendBtn.click();
        }
      });
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html)

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    answer, documents = session.send(
        payload.query,
        k=payload.top_k,
        temperature=payload.temperature
    )
    return AskResponse(query=payload.query, answer=answer, documents=documents)

@app.post("/clear", response_model=ClearResponse)
def clear_history():
    session.clear_history()
    return ClearResponse(status="ok", message="conversation history cleared")
