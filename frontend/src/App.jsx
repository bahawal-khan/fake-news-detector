import { useEffect, useState } from "react";
import { checkHealth, predictArticle } from "./api";
import VerdictStamp from "./components/VerdictStamp";
import ConfidenceMeter from "./components/ConfidenceMeter";
import "./App.css";

const SAMPLE_REAL = {
  title: "City council approves annual infrastructure budget",
  text: "Members of the city council voted 6 to 1 on Tuesday to approve the annual infrastructure budget after three months of public hearings. Officials said the funding will prioritize road repairs and water system upgrades over the next fiscal year.",
};

const SAMPLE_FAKE = {
  title: "SHOCKING: Doctors HATE this one weird trick",
  text: "You won't believe what scientists don't want you to know!!! This miracle cure has been hidden from the public for years and the mainstream media refuses to cover it. Share this before it gets taken down!!!",
};

function App() {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    checkHealth()
      .then((data) => setApiStatus(data.status === "ok" ? "online" : "offline"))
      .catch(() => setApiStatus("offline"));
  }, []);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (text.trim().length < 10) {
      setError("Paste at least a sentence or two of article body text.");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const data = await predictArticle({ title, text });
      setResult(data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Couldn't reach the analysis service. Is the backend running?"
      );
    } finally {
      setLoading(false);
    }
  };

  const loadSample = (sample) => {
    setTitle(sample.title);
    setText(sample.text);
    setResult(null);
    setError("");
  };

  const handleClear = () => {
    setTitle("");
    setText("");
    setResult(null);
    setError("");
  };

  return (
    <div className="desk">
      <div className="desk__texture" aria-hidden="true" />

      <header className="masthead">
        <div className="masthead__row">
          <span className="eyebrow">Case File · No. 001</span>
          <span className={`status-chip status-chip--${apiStatus}`}>
            <span className="status-chip__dot" />
            {apiStatus === "online" && "Analyst online"}
            {apiStatus === "offline" && "Analyst offline"}
            {apiStatus === "checking" && "Connecting…"}
          </span>
        </div>
        <h1 className="masthead__title">The Verification Desk</h1>
        <p className="masthead__subtitle">
          Submit a headline and body text. A classifier trained on TF-IDF
          textual patterns and handcrafted linguistic signals will render
          its verdict — <em>Real</em> or <em>Fake</em> — with a confidence
          score.
        </p>
      </header>

      <main className="desk__grid">
        <section className="intake" aria-label="Submit article for analysis">
          <form onSubmit={handleSubmit}>
            <label className="field">
              <span className="field__label">Headline <em>(optional)</em></span>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. Local hospital reports record vaccination turnout"
                maxLength={300}
              />
            </label>

            <label className="field">
              <span className="field__label">
                Article body
                <em>{wordCount > 0 ? `${wordCount} words` : "required"}</em>
              </span>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste the full article text here…"
                rows={10}
              />
            </label>

            {error && <p className="field__error">{error}</p>}

            <div className="intake__actions">
              <button type="submit" className="btn btn--primary" disabled={loading}>
                {loading ? "Examining…" : "Investigate"}
              </button>
              <button type="button" className="btn btn--ghost" onClick={handleClear}>
                Clear
              </button>
            </div>
          </form>

          <div className="samples">
            <span>Try an example:</span>
            <button type="button" onClick={() => loadSample(SAMPLE_REAL)}>
              Plausible article
            </button>
            <button type="button" onClick={() => loadSample(SAMPLE_FAKE)}>
              Sensational article
            </button>
          </div>
        </section>

        <section className="verdict-panel" aria-label="Analysis result" aria-live="polite">
          {!result && !loading && (
            <div className="verdict-panel__empty">
              <div className="empty-mark">?</div>
              <p>No case submitted yet.</p>
              <p className="empty-mark__hint">
                The verdict, stamped in ink, will appear here.
              </p>
            </div>
          )}

          {loading && (
            <div className="verdict-panel__loading">
              <div className="loading-spinner" />
              <p>Cross-referencing linguistic evidence…</p>
            </div>
          )}

          {result && !loading && (
            <div className="verdict-result">
              <VerdictStamp isFake={result.is_fake} />

              <div className="verdict-result__meta">
                <ConfidenceMeter
                  fakeProbability={result.fake_probability}
                  realProbability={result.real_probability}
                />

                <dl className="verdict-facts">
                  <div>
                    <dt>Confidence</dt>
                    <dd>{(result.confidence * 100).toFixed(1)}%</dd>
                  </div>
                  <div>
                    <dt>Model</dt>
                    <dd>{result.model_name}</dd>
                  </div>
                </dl>
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="colophon">
        <p>
          Classic ML pipeline — TF-IDF (1–2 grams) + handcrafted linguistic
          features (sentiment, punctuation density, length) combined and fed
          to a Linear SVM. No deep learning, no LLM calls at inference time.
        </p>
        <p className="colophon__stack">
          scikit-learn · FastAPI · React · trained on the Fake &amp; Real News
          dataset (Kaggle)
        </p>
      </footer>
    </div>
  );
}

export default App;
