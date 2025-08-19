# NBA Analytics Bot

Named AnalyiXpert...

A fullstack web app with a **React frontend** and a **FastAPI backend**.

* **Frontend**: [https://nba-analytics-bot-frontend.onrender.com](https://nba-analytics-bot-frontend.onrender.com)
* **Backend**: [https://nba-analytics-bot.onrender.com](https://nba-analytics-bot.onrender.com)

---

## 🚀 Project Structure

```
.
├── backend      # FastAPI backend (Uvicorn server)
├── frontend     # React (Vite) frontend
├── datasets     # Example datasets
└── README.md
```

---

## ⚙️ Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file inside `backend/`:

```
OPENAI_API_KEY=your_api_key_here
FRONTEND_ORIGIN=http://localhost:5173
```

Run:

```bash
uvicorn app.main:app --reload --port 8000
```

Check: [http://localhost:8000/health](http://localhost:8000/health)

---

### 3. Frontend

```bash
cd ../frontend
npm install
```

Create a `.env` file inside `frontend/`:

```
VITE_API_BASE_URL=http://localhost:8000
```

Run:

```bash
npm run dev
```

Visit: [http://localhost:5173](http://localhost:5173)

---

## 🔐 Environment Variables

| Service  | Variable            | Purpose             |
| -------- | ------------------- | ------------------- |
| Backend  | `OPENAI_API_KEY`    | API key             |
| Backend  | `FRONTEND_ORIGIN`   | Frontend URL (CORS) |
| Frontend | `VITE_API_BASE_URL` | Backend base URL    |

---

## ✅ Health Check

Backend endpoint:

```
GET /health
Response: { "ok": true }
```
