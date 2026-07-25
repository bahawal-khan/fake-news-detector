export default function ConfidenceMeter({ fakeProbability, realProbability }) {
  const fakePct = (fakeProbability * 100).toFixed(1);
  const realPct = (realProbability * 100).toFixed(1);

  return (
    <div className="meter" role="img" aria-label={`${fakePct}% fake, ${realPct}% real`}>
      <div className="meter__track">
        <div className="meter__fill meter__fill--fake" style={{ width: `${fakePct}%` }} />
      </div>
      <div className="meter__labels">
        <span className="meter__labels-fake">Fake {fakePct}%</span>
        <span className="meter__labels-real">Real {realPct}%</span>
      </div>
    </div>
  );
}
