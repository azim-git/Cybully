const EXAMPLES = [
  { label: "normal",     text: "yea, idk if I that's a good idea. i went there once and it stunk like crazy!" },
  { label: "offensive",  text: "fuck you and your big ass forehead. dickheaded cunt" },
  { label: "hatespeech", text: "all niggers are the same... get out the country" },
]

const LABEL_CONFIG = {
  normal:     { colour: "#22c55e" },
  offensive:  { colour: "#f97316" },
  hatespeech: { colour: "#ef4444" },
}

export default function ExamplePosts({ onSelect }) {
  return (
    <div className="examples">
      <h3>Try an example</h3>
      <div className="examples-row">
        {EXAMPLES.map((ex, i) => (
          <button
            key={i}
            className="example-btn"
            style={{ borderColor: LABEL_CONFIG[ex.label].colour }}
            onClick={() => onSelect(ex.text)}
          >
            <span
              className="example-label"
              style={{ color: LABEL_CONFIG[ex.label].colour }}
            >
              {ex.label}
            </span>
            <span className="example-text">{ex.text}</span>
          </button>
        ))}
      </div>
    </div>
  )
}