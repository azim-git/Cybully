const LABEL_CONFIG = {
  normal:     { colour: "#22c55e", description: "This post appears to be normal content." },
  offensive:  { colour: "#f97316", description: "This post contains offensive language." },
  hatespeech: { colour: "#ef4444", description: "This post contains hate speech." },
}

export default function ResultCard({ result }) {
  const config = LABEL_CONFIG[result.label]

  return (
    <div className="card result-card">
      <div className="label-badge" style={{ borderColor: config.colour, color: config.colour }}>
        <span className="label-text">{result.label.toUpperCase()}</span>
        <span className="label-confidence">{(result.confidence * 100).toFixed(1)}% confident</span>
      </div>

      <p className="label-description">{config.description}</p>

      <div className="scores">
        <h3>Confidence Scores</h3>
        {Object.entries(result.scores).map(([label, score]) => (
          <div key={label} className="score-row">
            <span className="score-label">{label}</span>
            <div className="score-bar-track">
              <div
                className="score-bar-fill"
                style={{
                  width: `${(score * 100).toFixed(1)}%`,
                  backgroundColor: LABEL_CONFIG[label].colour
                }}
              />
            </div>
            <span className="score-value">{(score * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      <p className="model-used">Model: {result.model_used}</p>
    </div>
  )
}