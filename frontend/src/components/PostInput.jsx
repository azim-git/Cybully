export default function PostInput({
  text, setText, models, modelName, setModelName, onSubmit, loading
}) {
  return (
    <div className="card input-card">
      <textarea
        className="textarea"
        rows={5}
        placeholder="Type or paste a social media post here..."
        value={text}
        onChange={e => setText(e.target.value)}
      />

      <div className="input-row">
        <div className="model-selector">
          <label htmlFor="model-select">Model</label>
          <select
            id="model-select"
            value={modelName}
            onChange={e => setModelName(e.target.value)}
          >
            {models.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        <button
          className="submit-btn"
          onClick={onSubmit}
          disabled={loading || !text.trim()}
        >
          {loading ? "Analysing..." : "Analyse Post"}
        </button>
      </div>
    </div>
  )
}