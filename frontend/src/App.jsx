import { useState, useEffect } from "react"
import axios from "axios"
import PostInput from "./components/PostInput"
import ResultCard from "./components/ResultCard"
import ExamplePosts from "./components/ExamplePosts"
import "./App.css"

const API_BASE = "http://localhost:8000"

export default function App() {
  const [models, setModels]       = useState([])
  const [text, setText]           = useState("")
  const [modelName, setModelName] = useState("")
  const [result, setResult]       = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)

  // fetch available models on mount
  useEffect(() => {
    axios.get(`${API_BASE}/models`)
      .then(res => {
        setModels(res.data.models)
        setModelName(res.data.models[0])   // default to first model
      })
      .catch(() => setError("Could not reach the API. Is the server running?"))
  }, [])

  const handleSubmit = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await axios.post(`${API_BASE}/predict`, {
        text,
        model_name: modelName
      })
      setResult(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || "Something went wrong.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Cyberbullying Classifier</h1>
        <p>Analyse social media posts for harmful content using fine-tuned DistilBERT.</p>
      </header>

      <main className="main">
        <ExamplePosts onSelect={setText} />

        <PostInput
          text={text}
          setText={setText}
          models={models}
          modelName={modelName}
          setModelName={setModelName}
          onSubmit={handleSubmit}
          loading={loading}
        />

        {error && <div className="error">{error}</div>}

        {result && <ResultCard result={result} />}
      </main>

      <footer className="disclaimer">
        <strong>Disclaimer:</strong> The example posts are provided solely to demonstrate the model's capabilities. The creator does not endorse or agree with any of the example content.
      </footer>
    </div>
  )
}