export default function VerdictStamp({ isFake }) {
  const label = isFake ? "FAKE" : "REAL";

  return (
    <div className={`stamp stamp--${isFake ? "fake" : "real"}`}>
      <svg viewBox="0 0 220 220" className="stamp__ring" aria-hidden="true">
        <circle cx="110" cy="110" r="98" />
        <text>
          <textPath href="#stampCirclePath">
            {isFake ? "· UNVERIFIED · UNVERIFIED · " : "· VERIFIED · VERIFIED · "}
          </textPath>
        </text>
        <path
          id="stampCirclePath"
          d="M 110,110 m -80,0 a 80,80 0 1,1 160,0 a 80,80 0 1,1 -160,0"
          fill="none"
        />
      </svg>
      <span className="stamp__label">{label}</span>
    </div>
  );
}
